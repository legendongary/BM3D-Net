#include "mex.h"
#include "gpu/mxGPUArray.h"

/*kernel function of calculating derivates*/
void __global__ calDzDc(const double *Coff, const double *Derv, const double *Norm, double *DzDc, int N, int K)
{
    // Coff: cofficients of size NxK
    // N: norm of size 1xK
    // Derv: derivate of size NxK
    // M: size of one cofficient
    // N: number of basises
    // K: number of filters
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(; k<K; k+=total_threads)
    {
        double SUM = 0;
        for(int n=0; n<N; n++)
        {
            SUM += Coff[k*N+n] * Derv[k*N+n];
        }
        double temp = 1 / Norm[k];
        for(int n=0; n<N; n++)
        {
            DzDc[k*N+n] = temp * (Derv[k*N+n] -  Coff[k*N+n] * SUM);
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    mxGPUArray const *Coff;
    mxGPUArray const *Derv;
    mxGPUArray const *Norm;
    double const *dCoff;
    double const *dDerv;
    double const *dNorm;
    
    mxGPUArray *DzDc;
    double *dDzDc;
    
    int N, K;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    mxInitGPU();
    
    if ((nrhs!=3) || !(mxIsGPUArray(prhs[0])))
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    Coff = mxGPUCreateFromMxArray(prhs[0]);
    Derv = mxGPUCreateFromMxArray(prhs[1]);
    Norm = mxGPUCreateFromMxArray(prhs[2]);
    if(mxGPUGetClassID(Coff) != mxDOUBLE_CLASS || mxGPUGetClassID(Derv) != mxDOUBLE_CLASS || mxGPUGetClassID(Derv) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    dCoff = (const double *)(mxGPUGetDataReadOnly(Coff));
    dDerv = (const double *)(mxGPUGetDataReadOnly(Derv));
    dNorm = (const double *)(mxGPUGetDataReadOnly(Norm));
    
    const mwSize *dimC = mxGPUGetDimensions(Coff);
    N = dimC[0];
    K = dimC[1];
    
    DzDc = mxGPUCreateGPUArray(2, dimC, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    dDzDc = (double *)(mxGPUGetData(DzDc));
    
    blocksPerGrid = (K + threadsPerBlock - 1) / threadsPerBlock;
    calDzDc<<<blocksPerGrid, threadsPerBlock>>>(dCoff, dDerv, dNorm, dDzDc, N, K);
    plhs[0] = mxGPUCreateMxArrayOnGPU(DzDc);
    
    mxGPUDestroyGPUArray(Coff);
    mxGPUDestroyGPUArray(Derv);
    mxGPUDestroyGPUArray(Norm);
    mxGPUDestroyGPUArray(DzDc);
}