// Explicit CUDA conditional graph around separately captured TensorRT engines.
// No TensorRT If capture, host callbacks, CPU choice readback or run-both select.
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <exception>

static thread_local std::string last_error;
#define API extern "C" __declspec(dllexport)
static void checked_cuda(cudaError_t e, const char *expression, int line) {
    if (e != cudaSuccess) throw std::string(expression) + " at line " +
        std::to_string(line) + ": " + cudaGetErrorName(e) + " (" +
        std::to_string((int)e) + ") " + cudaGetErrorString(e);
}
#define ck(expression) checked_cuda((expression), #expression, __LINE__)
struct State {
    cudaGraph_t graph=nullptr, body[2]{};
    cudaGraphExec_t executable=nullptr;
    cudaGraphConditionalHandle handles[2]{};
    cudaGraphNode_t conditions[2]{};
    const float *prob=nullptr, *branch_output[2]{};
    float *output=nullptr;
    int *choice=nullptr, *counts=nullptr, *prediction=nullptr;
    int ia=0, ib=0, indices[2]{0,1};
    float oa=0, ob=0, bias=0;
};
__global__ void set_conditions(const float *q, int ia, int ib, float oa,
                               float ob, float bias,
                               cudaGraphConditionalHandle a,
                               cudaGraphConditionalHandle b) {
    bool take_a=__fsub_rn(__fsub_rn(q[ia],oa),__fsub_rn(q[ib],ob)) >= bias;
    cudaGraphSetConditional(a, take_a ? 1u : 0u);
    cudaGraphSetConditional(b, take_a ? 0u : 1u);
}
__global__ void copy_selected(const float *source, float *out, int *choice,
                              int *counts, int selected) {
    if(threadIdx.x<100) out[threadIdx.x]=source[threadIdx.x];
    if(threadIdx.x==0) { *choice=selected; atomicAdd(counts+selected,1); }
}
__global__ void predict(const float *out, int *prediction) {
    int best=0;
    for(int i=1;i<100;++i) if(out[i]>out[best]) best=i;
    *prediction=best;
}
static std::vector<cudaGraphNode_t> leaves(cudaGraph_t graph) {
    size_t count=0; ck(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count), result;
    ck(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes) {
        size_t children=0; ck(cudaGraphNodeGetDependentNodes(node,nullptr,&children));
        if(!children) result.push_back(node);
    }
    return result;
}
template<class F> int wrapped(F fn) {
    try { fn(); return 0; }
    catch(const std::string &e) { last_error=e; return -1; }
    catch(const std::exception &e) { last_error=e.what(); return -2; }
    catch(...) { last_error="Unknown native error"; return -3; }
}
API const char *graph_last_error() { return last_error.c_str(); }
API int graph_create(void **out, const float *prob, const float *a, const float *b,
                     float *output, int *choice, int *counts, int *prediction,
                     int ia,int ib,float oa,float ob,float bias) {
    return wrapped([&] {
        State *s=new State(); *out=s;
        s->prob=prob; s->branch_output[0]=a; s->branch_output[1]=b;
        s->output=output; s->choice=choice; s->counts=counts; s->prediction=prediction;
        s->ia=ia; s->ib=ib; s->oa=oa; s->ob=ob; s->bias=bias;
        ck(cudaGraphCreate(&s->graph,0));
        ck(cudaGraphConditionalHandleCreate(&s->handles[0],s->graph,0,cudaGraphCondAssignDefault));
        ck(cudaGraphConditionalHandleCreate(&s->handles[1],s->graph,0,cudaGraphCondAssignDefault));
    });
}
API int graph_begin_router(void *p, void *stream) {
    return wrapped([&] { auto s=(State*)p;
        ck(cudaStreamBeginCaptureToGraph((cudaStream_t)stream,s->graph,nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
    });
}
API int graph_end_router(void *p, void *stream) {
    return wrapped([&] {
        auto s=(State*)p; cudaGraph_t captured;
        ck(cudaStreamEndCapture((cudaStream_t)stream,&captured));
        if(captured!=s->graph) throw std::string("Wrong router capture graph");
        auto deps=leaves(s->graph);
        if(deps.empty()) throw std::string("Empty captured router");
        void *args[]={&s->prob,&s->ia,&s->ib,&s->oa,&s->ob,&s->bias,&s->handles[0],&s->handles[1]};
        cudaKernelNodeParams k{}; k.func=(void*)set_conditions;
        k.gridDim=dim3(1); k.blockDim=dim3(1); k.kernelParams=args;
        cudaGraphNode_t setter;
        ck(cudaGraphAddKernelNode(&setter,s->graph,deps.data(),deps.size(),&k));
        for(int i=0;i<2;++i) {
            cudaGraphNodeParams c{}; c.type=cudaGraphNodeTypeConditional;
            c.conditional.handle=s->handles[i]; c.conditional.type=cudaGraphCondTypeIf;
            c.conditional.size=1;
            ck(cudaGraphAddNode(&s->conditions[i],s->graph,&setter,1,&c));
            s->body[i]=c.conditional.phGraph_out[0];
        }
    });
}
API int graph_begin_body(void *p, void *stream, int i) {
    return wrapped([&] { auto s=(State*)p;
        if(i<0 || i>1) throw std::string("Invalid branch");
        ck(cudaStreamBeginCaptureToGraph((cudaStream_t)stream,s->body[i],nullptr,nullptr,0,cudaStreamCaptureModeGlobal));
    });
}
API int graph_end_body(void *p, void *stream, int i) {
    return wrapped([&] {
        auto s=(State*)p; cudaGraph_t captured;
        ck(cudaStreamEndCapture((cudaStream_t)stream,&captured));
        if(captured!=s->body[i]) throw std::string("Wrong body graph");
        auto deps=leaves(s->body[i]);
        if(deps.empty()) throw std::string("Empty captured classifier");
        void *args[]={&s->branch_output[i],&s->output,&s->choice,&s->counts,&s->indices[i]};
        cudaKernelNodeParams k{}; k.func=(void*)copy_selected;
        k.gridDim=dim3(1); k.blockDim=dim3(128); k.kernelParams=args;
        cudaGraphNode_t marker;
        ck(cudaGraphAddKernelNode(&marker,s->body[i],deps.data(),deps.size(),&k));
    });
}
API int graph_finish(void *p,const char *dot) {
    return wrapped([&] {
        auto s=(State*)p;
        void *args[]={&s->output,&s->prediction};
        cudaKernelNodeParams k{}; k.func=(void*)predict;
        k.gridDim=dim3(1); k.blockDim=dim3(1); k.kernelParams=args;
        cudaGraphNode_t final;
        ck(cudaGraphAddKernelNode(&final,s->graph,s->conditions,2,&k));
        ck(cudaGraphDebugDotPrint(s->graph,dot,cudaGraphDebugDotFlagsVerbose));
        // Inspect source/body structure before instantiation in this revision.
    });
}
API int graph_instantiate(void *p) {
    return wrapped([&] { auto s=(State*)p;
        ck(cudaGraphInstantiate(&s->executable,s->graph,0));
    });
}
API int graph_launch(void *p,void *stream) {
    return wrapped([&] { ck(cudaGraphLaunch(((State*)p)->executable,(cudaStream_t)stream)); });
}
API int graph_node_counts(void *p,int *counts) {
    return wrapped([&] {
        auto s=(State*)p;
        for(int g=0;g<3;++g) {
            auto graph=g==0?s->graph:s->body[g-1];
            size_t n=0;
            try { ck(cudaGraphGetNodes(graph,nullptr,&n)); }
            catch(const std::string &e) { throw std::string("graph index ")+std::to_string(g)+": "+e; }
            std::vector<cudaGraphNode_t> nodes(n); ck(cudaGraphGetNodes(graph,nodes.data(),&n));
            counts[g*3]=(int)n; counts[g*3+1]=0; counts[g*3+2]=0;
            for(auto node:nodes) {
                cudaGraphNodeType type;
                try { ck(cudaGraphNodeGetType(node,&type)); }
                catch(const std::string &e) { throw std::string("node type graph ")+std::to_string(g)+": "+e; }
                if(type==cudaGraphNodeTypeHost) counts[g*3+1]++;
                if(type==cudaGraphNodeTypeConditional) counts[g*3+2]++;
            }
        }
    });
}
API int graph_destroy(void *p) {
    return wrapped([&] { auto s=(State*)p;
        if(s->executable) ck(cudaGraphExecDestroy(s->executable));
        if(s->graph) ck(cudaGraphDestroy(s->graph));
        delete s;
    });
}
