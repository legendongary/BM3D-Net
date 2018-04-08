#include "mex.h"
#include "gpu/mxGPUArray.h"

void __global__ norm_ker(const double *vec, double *nvec, double *norm, int N, int L)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(; n<N; n+=total_threads)
    {
        double temp;
        for(int l=0; l<L; l++)
        {
            temp = vec[n*L+l] * vec[n*L+l];
            norm[n] = norm[n] + temp;
        }
        norm[n] = sqrt(norm[n]);
        temp =1 / norm[n];
        for(int l=0; l<L; l++)
        {
            nvec[n*L+l] = vec[n*L+l] * temp;
            
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    mxGPUArray const *filter;
    double const * d_filter;
    
    mxGPUArray *nfilter;
    mxGPUArray *norm;
    double *d_nfilter;
    double *d_norm;
    
    int N, L;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    mxInitGPU();
    
    if ((nrhs!=1) || !(mxIsGPUArray(prhs[0])))
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    filter = mxGPUCreateFromMxArray(prhs[0]);
    if(mxGPUGetClassID(filter) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    d_filter = (const double *)(mxGPUGetDataReadOnly(filter));
    
    const mwSize *dimf = mxGPUGetDimensions(filter);
    L = dimf[0];
    N = dimf[1];
    nfilter = mxGPUCreateGPUArray(2, dimf, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    d_nfilter = (double *)(mxGPUGetData(nfilter));
    mwSize dimn[2];
    dimn[0] = 1;
    dimn[1] = N;
    norm = mxGPUCreateGPUArray(2, dimn, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    d_norm = (double *)(mxGPUGetData(norm));
    
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    norm_ker<<<blocksPerGrid, threadsPerBlock>>>(d_filter, d_nfilter, d_norm, N, L);
    plhs[0] = mxGPUCreateMxArrayOnGPU(nfilter);
    plhs[1] = mxGPUCreateMxArrayOnGPU(norm);
    
    mxGPUDestroyGPUArray(filter);
    mxGPUDestroyGPUArray(nfilter);
    mxGPUDestroyGPUArray(norm);
    
}