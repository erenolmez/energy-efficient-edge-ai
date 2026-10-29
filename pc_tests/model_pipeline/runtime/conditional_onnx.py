"""Compose actual ONNX If branches around classifier graphs, not Where outputs."""
import copy
import numpy as np

def fixed_model(model):
    model=copy.deepcopy(model)
    for value in list(model.graph.input)+list(model.graph.output):
        if value.type.tensor_type.shape.dim:
            value.type.tensor_type.shape.dim[0].ClearField('dim_param')
            value.type.tensor_type.shape.dim[0].dim_value=1
    return model

def branch_graph(model,prefix):
    graph=copy.deepcopy(fixed_model(model).graph)
    if len(graph.input)!=1 or graph.input[0].name!='input':raise ValueError('One captured normalized input required')
    def rename(name):return name if name in ('','input') else prefix+name
    for node in graph.node:
        node.name=prefix+(node.name or node.op_type)
        node.input[:]=[rename(n) for n in node.input];node.output[:]=[rename(n) for n in node.output]
    for value in list(graph.initializer)+list(graph.value_info)+list(graph.output):value.name=rename(value.name)
    del graph.input[:]
    graph.name=prefix+'branch'
    return graph

def make_conditional(router,a,b,columns,costs,penalty,bias=0.):
    import onnx
    from onnx import helper as h,numpy_helper as nh,TensorProto as T
    model=copy.deepcopy(router);graph=model.graph
    if len(graph.input)!=1 or graph.input[0].name!='input' or len(graph.output)!=1:raise ValueError('Expected fixed oneinput router')
    probabilities=graph.output[0].name
    constants={'idx_a':np.array(columns[0],dtype=np.int64),'idx_b':np.array(columns[1],dtype=np.int64),
        'cost_a':np.array(penalty*costs[0]/max(costs),dtype=np.float32),
        'cost_b':np.array(penalty*costs[1]/max(costs),dtype=np.float32),'axes0':np.array([0],dtype=np.int64),
        'bias':np.array(bias,dtype=np.float32)}
    graph.initializer.extend([nh.from_array(v,'dispatch_'+n) for n,v in constants.items()])
    graph.node.extend([
        h.make_node('Gather',[probabilities,'dispatch_idx_a'],['dispatch_qa'],axis=1,name='dispatch_gather_a'),
        h.make_node('Gather',[probabilities,'dispatch_idx_b'],['dispatch_qb'],axis=1,name='dispatch_gather_b'),
        h.make_node('Sub',['dispatch_qa','dispatch_cost_a'],['dispatch_sa']),
        h.make_node('Sub',['dispatch_qb','dispatch_cost_b'],['dispatch_sb']),
        h.make_node('Sub',['dispatch_sa','dispatch_sb'],['dispatch_difference']),
        h.make_node('GreaterOrEqual',['dispatch_difference','dispatch_bias'],['dispatch_condition_vector']),
        h.make_node('Squeeze',['dispatch_condition_vector','dispatch_axes0'],['dispatch_condition']),
        h.make_node('If',['dispatch_condition'],['logits'],name='dispatch_native_if',then_branch=branch_graph(a,'A__'),else_branch=branch_graph(b,'B__')),
        h.make_node('Not',['dispatch_condition'],['dispatch_not_condition']),
        h.make_node('Cast',['dispatch_not_condition'],['dispatch_choice_scalar'],to=T.INT32),
        h.make_node('Unsqueeze',['dispatch_choice_scalar','dispatch_axes0'],['choice'])])
    del graph.output[:]
    graph.output.extend([h.make_tensor_value_info('logits',T.FLOAT,[1,100]),h.make_tensor_value_info('choice',T.INT32,[1])])
    model.ir_version=min(model.ir_version,8)
    onnx.checker.check_model(model)
    return model
