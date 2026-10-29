import unittest
import numpy as np

class ConditionalGraphTests(unittest.TestCase):
    def test_true_if_lexical_input_and_output_id(self):
        import onnx
        from onnx import helper as h,TensorProto as T,numpy_helper as nh
        from onnx.reference import ReferenceEvaluator
        from pc_tests.model_pipeline.runtime.conditional_onnx import make_conditional
        inp=h.make_tensor_value_info('input',T.FLOAT,[1,3,2,2])
        r=h.make_graph([h.make_node('ReduceMean',['input'],['p'],axes=[1,2,3],keepdims=0),h.make_node('Sub',['one','p'],['op']),
            h.make_node('Unsqueeze',['p','axes'],['pp']),h.make_node('Unsqueeze',['op','axes'],['oo']),h.make_node('Concat',['pp','oo'],['q'],axis=1)],
            'router',[inp],[h.make_tensor_value_info('q',T.FLOAT,[1,2])],[nh.from_array(np.array(1,np.float32),'one'),nh.from_array(np.array([1],np.int64),'axes')])
        def branch(value):
            return h.make_model(h.make_graph([h.make_node('Flatten',['input'],['flat']),h.make_node('Gemm',['flat','w','b'],['output'])],
                'classifier',[inp],[h.make_tensor_value_info('output',T.FLOAT,[1,100])],
                [nh.from_array(np.zeros((12,100),np.float32),'w'),nh.from_array(np.full(100,value,np.float32),'b')]),opset_imports=[h.make_opsetid('',17)])
        router=h.make_model(r,opset_imports=[h.make_opsetid('',17)])
        model=make_conditional(router,branch(1),branch(2),[0,1],[1.,1.],0.)
        evaluator=ReferenceEvaluator(model)
        for val,choice,logit in [(.8,0,1),(.2,1,2)]:
            result=evaluator.run(None,{'input':np.full((1,3,2,2),val,np.float32)})
            self.assertTrue(np.all(result[0]==logit));self.assertEqual(int(result[1][0]),choice)
        self.assertEqual(sum(n.op_type=='If' for n in model.graph.node),1)
        self.assertFalse(any(n.op_type=='Where' for n in model.graph.node))

if __name__=='__main__':unittest.main()
