/* CUDA program for Gaussian Kernel Regression */

#include "mex.h"
#include "math.h"
#include "gpu/mxGPUArray.h"

/* evaluate gaussian kernel regression kernel */
__global__ void gkr_eval_ker(double * const yvar, const double * const xvar, const double * const mu, const double * const gamma, const int M, const int N)
{
    int T = M * N;
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    double ga = 1 / (gamma[0] * gamma[0] * 2);
    for(; t < T; t += total_threads)
    {
        int m = t % M;
        int n = (t - m) / M;
        double va = xvar[m];
        double me = mu[n];
        yvar[t] = exp(- (va - me) * (va - me) * ga);
    }
}
/* calculate gradient of gaussian */
__global__ void gkr_grad_ker(double * const grad, const double * const xvar, const double * const yvar, const double * const mu, const double * const gamma, const int M, const int N)
{
    int T = M * N;
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    double ga = 1 / (gamma[0] * gamma[0]);
    for(; t < T; t += total_threads)
    {
        int m = t % M;
        int n = (t - m) / M;
        grad[t] = - yvar[t] * (xvar[m] - mu[n]) * ga;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    
    if(nrhs == 3)
    {
        mxGPUArray const *xvar;
        mxGPUArray const *mu;
        mxGPUArray const *gamma;
        mxGPUArray       *yvar;
        
        double const *d_xvar;
        double const *d_mu;
        double const *d_gamma;
        double       *d_yvar;
        
        int const threadsPerBlock = 1024;
        int blocksPerGrid;
        
        mxInitGPU();
        
        xvar  = mxGPUCreateFromMxArray(prhs[0]);
        mu    = mxGPUCreateFromMxArray(prhs[1]);
        gamma = mxGPUCreateFromMxArray(prhs[2]);
        
        int M = (int)(mxGPUGetNumberOfElements(xvar));
        int N = (int)(mxGPUGetNumberOfElements(mu));
        
        d_xvar  = (double const *)(mxGPUGetDataReadOnly(xvar));
        d_mu    = (double const *)(mxGPUGetDataReadOnly(mu));
        d_gamma = (double const *)(mxGPUGetDataReadOnly(gamma));
        
        mwSize yvdim[2];
        yvdim[0] = M;
        yvdim[1] = N;
        yvar = mxGPUCreateGPUArray(2, yvdim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
        d_yvar = (double *)(mxGPUGetData(yvar));
        
        blocksPerGrid = (M * N + threadsPerBlock - 1) / threadsPerBlock;
        gkr_eval_ker<<<blocksPerGrid, threadsPerBlock>>>(d_yvar, d_xvar, d_mu, d_gamma, M, N);
        plhs[0] = mxGPUCreateMxArrayOnGPU(yvar);
        
        mxGPUDestroyGPUArray(yvar);
        mxGPUDestroyGPUArray(xvar);
        mxGPUDestroyGPUArray(mu);
        mxGPUDestroyGPUArray(gamma);
    }
    if(nrhs == 4)
    {
        
        mxGPUArray const *xvar;
        mxGPUArray const *yvar;
        mxGPUArray const *mu;
        mxGPUArray const *gamma;
        mxGPUArray       *grad;
        
        double const *d_xvar;
        double const *d_mu;
        double const *d_gamma;
        double const *d_yvar;
        double       *d_grad;
        
        int const threadsPerBlock = 1024;
        int blocksPerGrid;
        
        mxInitGPU();
        
        xvar  = mxGPUCreateFromMxArray(prhs[0]);
        yvar  = mxGPUCreateFromMxArray(prhs[1]);
        mu    = mxGPUCreateFromMxArray(prhs[2]);
        gamma = mxGPUCreateFromMxArray(prhs[3]);
        
        int M = (int)(mxGPUGetNumberOfElements(xvar));
        int N = (int)(mxGPUGetNumberOfElements(mu));
        
        d_xvar  = (double const *)(mxGPUGetDataReadOnly(xvar));
        d_yvar  = (double const *)(mxGPUGetDataReadOnly(yvar));
        d_mu    = (double const *)(mxGPUGetDataReadOnly(mu));
        d_gamma = (double const *)(mxGPUGetDataReadOnly(gamma));
        
        mwSize grdim[2];
        grdim[0] = M;
        grdim[1] = N;
        grad = mxGPUCreateGPUArray(2, grdim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
        d_grad  = (double *)(mxGPUGetData(grad));
        
        blocksPerGrid = (M * N + threadsPerBlock - 1) / threadsPerBlock;
        gkr_grad_ker<<<blocksPerGrid, threadsPerBlock>>>(d_grad, d_xvar, d_yvar, d_mu, d_gamma, M, N);
        plhs[0] = mxGPUCreateMxArrayOnGPU(grad);
        
        mxGPUDestroyGPUArray(grad);
        mxGPUDestroyGPUArray(yvar);
        mxGPUDestroyGPUArray(xvar);
        mxGPUDestroyGPUArray(mu);
        mxGPUDestroyGPUArray(gamma);
    }
}