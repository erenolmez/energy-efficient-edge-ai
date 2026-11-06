// Eight standalone TensorRT engines in explicit CUDA IF bodies.
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

constexpr int N = 8;
#define API extern "C" __declspec(dllexport)
static thread_local std::string error_text;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
}
template<class F> int checked(F f) {
    try { f(); return 0; }
    catch (const std::string& e) { error_text=e; return -1; }
    catch (...) { error_text="Native graph error"; return -2; }
}
struct Rule {
    cudaGraphConditionalHandle h[N]{};
    int column[N]{};
    float offset[N]{};
};
struct State {
    cudaGraph_t parent=nullptr, bodies[N]{};
    cudaGraphExec_t exec=nullptr;
    Rule rule{};
    const float *prob=nullptr, *logits=nullptr;
    int *choice=nullptr, *pred=nullptr, *counts=nullptr;
    int batch=0;
};
__global__ void choose(const float *prob, Rule rule, int *choice) {
    int best=0;
    float score=__fsub_rn(prob[rule.column[0]],rule.offset[0]);
    for (int i=1; i<N; ++i) {
        float next=__fsub_rn(prob[rule.column[i]],rule.offset[i]);
        if (next>score) { best=i; score=next; }
    }
    *choice=best;
    for (int i=0; i<N; ++i) cudaGraphSetConditional(rule.h[i], i==best);
}
__global__ void finish_branch(const float *logits, int *pred, int *counts, int branch, int batch) {
    int row=threadIdx.x;
    if (row<batch) {
        const float *p=logits+100*row;
        int best=0;
        for (int j=1;j<100;++j) if (p[j]>p[best]) best=j;
        pred[row]=best;
    }
    if (row==0) ++counts[branch];
}
static std::vector<cudaGraphNode_t> leaves(cudaGraph_t g) {
    size_t n=0; ck(cudaGraphGetNodes(g,nullptr,&n));
    std::vector<cudaGraphNode_t> nodes(n), out;
    ck(cudaGraphGetNodes(g,nodes.data(),&n));
    for (auto node:nodes) {
        size_t children=0; ck(cudaGraphNodeGetDependentNodes(node,nullptr,&children));
        if (!children) out.push_back(node);
    }
    if (out.empty()) throw std::string("Empty capture");
    return out;
}
static cudaGraphNode_t kernel(cudaGraph_t g, const std::vector<cudaGraphNode_t>& deps,
                              void *fn, int threads, void **args) {
    cudaKernelNodeParams p{};
    p.func=fn; p.gridDim=dim3(1); p.blockDim=dim3(threads); p.kernelParams=args;
    cudaGraphNode_t node;
    ck(cudaGraphAddKernelNode(&node,g,deps.data(),deps.size(),&p));
    return node;
}
static void verify(cudaGraph_t graph, int *counts) {
    size_t n=0; ck(cudaGraphGetNodes(graph,nullptr,&n));
    std::vector<cudaGraphNode_t> nodes(n);
    ck(cudaGraphGetNodes(graph,nodes.data(),&n));
    counts[0]+=(int)n;
    for (auto node:nodes) {
        CUgraphNodeType type;
        if (cuGraphNodeGetType(reinterpret_cast<CUgraphNode>(node),&type)!=CUDA_SUCCESS)
            throw std::string("Could not inspect CUDA graph node");
        if (type==CU_GRAPH_NODE_TYPE_HOST) ++counts[1];
        if (type==CU_GRAPH_NODE_TYPE_CONDITIONAL) ++counts[2];
        if (type==CU_GRAPH_NODE_TYPE_GRAPH) {
            cudaGraph_t child;
            ck(cudaGraphChildGraphNodeGetGraph(node,&child));
            verify(child,counts);
        }
    }
}
API const char *combined_error() { return error_text.c_str(); }
API int combined_create(void **p,const float *prob,const float *logits,int *choice,int *pred,
                        int *counts,int batch,const int *columns,const float *offsets) {
    return checked([&] {
        if (batch<1 || batch>128) throw std::string("Invalid batch");
        State *s=new State(); *p=s;
        s->prob=prob; s->logits=logits; s->choice=choice; s->pred=pred; s->counts=counts; s->batch=batch;
        ck(cudaGraphCreate(&s->parent,0));
        for(int i=0;i<N;++i) {
            s->rule.column[i]=columns[i]; s->rule.offset[i]=offsets[i];
            ck(cudaGraphConditionalHandleCreate(&s->rule.h[i],s->parent,0,cudaGraphCondAssignDefault));
        }
    });
}
API int combined_begin(void *p,void *stream,int branch) {
    return checked([&] {
        auto s=(State*)p;
        ck(cudaStreamBeginCaptureToGraph((cudaStream_t)stream,
            branch<0?s->parent:s->bodies[branch],nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
    });
}
API int combined_end(void *p,void *stream,int branch) {
    return checked([&] {
        auto s=(State*)p;
        cudaGraph_t graph;
        ck(cudaStreamEndCapture((cudaStream_t)stream,&graph));
        if(graph!=(branch<0?s->parent:s->bodies[branch])) throw std::string("Wrong capture graph");
        auto deps=leaves(graph);
        if (branch<0) {
            void *args[]={&s->prob,&s->rule,&s->choice};
            auto selector=kernel(graph,deps,(void*)choose,1,args);
            for(int i=0;i<N;++i) {
                cudaGraphNodeParams p{}; p.type=cudaGraphNodeTypeConditional;
                p.conditional.handle=s->rule.h[i]; p.conditional.type=cudaGraphCondTypeIf; p.conditional.size=1;
                cudaGraphNode_t node;
                ck(cudaGraphAddNode(&node,graph,&selector,1,&p));
                s->bodies[i]=p.conditional.phGraph_out[0];
            }
        } else {
            void *args[]={&s->logits,&s->pred,&s->counts,&branch,&s->batch};
            kernel(graph,deps,(void*)finish_branch,128,args);
        }
    });
}
API int combined_instantiate(void *p,int *stats) {
    return checked([&] {
        auto s=(State*)p;
        verify(s->parent,stats);
        if (stats[1] || stats[2]!=N) throw std::string("Unexpected parent graph structure");
        for(int i=0;i<N;++i) {
            int *entry=stats+3*(i+1);
            verify(s->bodies[i],entry);
            if (entry[0]<2 || entry[1] || entry[2]) throw std::string("Unexpected body structure");
        }
        ck(cudaGraphInstantiate(&s->exec,s->parent,0));
    });
}
API int combined_launch(void *p,void *stream) {
    return checked([&]{ck(cudaGraphLaunch(((State*)p)->exec,(cudaStream_t)stream));});
}
API int combined_destroy(void *p) {
    return checked([&] {
        auto s=(State*)p;
        if (s->exec) ck(cudaGraphExecDestroy(s->exec));
        if (s->parent) ck(cudaGraphDestroy(s->parent));
        delete s;
    });
}
