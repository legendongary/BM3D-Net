#include "mex.h"
#include <cmath>
#include "gpu/mxGPUArray.h"

#define IDX2(X, n1, n2, i1, i2) (X[(i2)*(n1) + (i1)])
#define IDX3(X, n1, n2, n3, i1, i2, i3) (X[(i3)*((n1)*(n2)) + (i2)*(n1) + (i1)])
#define IDX4(X, n1, n2, n3, n4, i1, i2, i3, i4) (X[(i4)*((n1)*(n2)*(n3)) + (i3)*((n1)*(n2)) + (i2)*(n1) + (i1)])

void __global__ nnagg_ker(
        const double *group, const double *index,
        double *storeIm, double *storeIn,
        int M, int N, int G, int P, int W)
{
    // M: image height
    // N: image width
    // G: group number
    // P: patch size
    // W: window size
    int hP = (P - 1) / 2;
    int hW = (W - 1) / 2;
    int T = M * N * G;
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(; t<T; t+=total_threads)
    {
        int idx = t % M;
        int idy = (t-idx) % (M*N) / M;
        int idz = (t-idy*M-idx) / (M*N);
       // if (idz == 1 & idx == 0 & idy == 0)
      //  {
            //int co = 0;
        double sum1 = 0;
        double sum2 = 0;
        for(int i=idx-hW; i<idx+hW; i++)
        {
            for(int j=idy-hW; j<idy+hW; j++)
            {
                if(i>=0 && i<M && j>=0 && j<N)
                {
                    int p = IDX4(index, M, N, G, 2, i, j, idz, 0) - 1;
                    int q = IDX4(index, M, N, G, 2, i, j, idz, 1) - 1;
                    int dp = idx - p;
                    int dq = idy - q;
                    if(dp<=hP && dp>=-hP && dq<=hP && dq >= -hP)
                    {
                        sum1 += IDX3(group, M, N, G*P*P, i, j, idz*P*P+(hP+dq)*P+(hP+dp));
                        //storeIm[co] = p + 1;//p q, dp, dqIDX3(group, M, N, G*P*P, p, q, idz*P*P+(hP+dq)*P+(hP+dp));
                        //storeIm[co + 1] = q + 1;//p q, dp, dqIDX3(group, M, N, G*P*P, p, q, idz*P*P+(hP+dq)*P+(hP+dp));
                        //storeIm[co + 2] = dp + hP;
                        //storeIm[co + 3] = dq + hP;
                        //storeIm[co + 4] = IDX3(group, M, N, G*P*P, i, j, idz*P*P+(hP+dq)*P+(hP+dp));
                        
                        sum2 += 1;
                    }
                }
            }
        }
        if(sum2==0)
        {
            sum2 = 1;
        }
        IDX3(storeIm, M, N, G, idx, idy, idz) = sum1;
        IDX3(storeIn, M, N, G, idx, idy, idz) = sum2;
    }
}

void __global__ nnsum_ker(
        const double *storeIm, const double *storeIn,
        double *image, double *overlap,
        int M, int N, int G)
{
    int T = M * N;
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(; t<T; t+=total_threads)
    {
        int idx = t % M;
        int idy = (t-idx) / M;
        double sum1 = 0;
        for(int i=0; i<G; i++)
        {
            sum1 +=  IDX3(storeIn, M, N, G, idx, idy, i);
        }
        IDX2(overlap, M, N, idx, idy) = sum1;
        
        double sum2 = 0;
        for(int i=0; i<G; i++)
        {
            sum2 += IDX3(storeIm, M, N, G, idx, idy, i);
        }
        //IDX2(image, M, N, idx, idy) = sum2;
        
        IDX2(image, M, N, idx, idy)  = sum2 / sum1; ///= IDX2(overlap, M, N, idx, idy);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    /**/
    mxGPUArray const *group;
    mxGPUArray const *index;
    double const *d_group;
    double const *d_index;
    mxGPUArray *storeIm;
    mxGPUArray *storeIn;
    mxGPUArray *image;
    mxGPUArray *overlap;
    int M, N, G, P, W;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    /**/
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /**/
    group = mxGPUCreateFromMxArray(prhs[0]);
    index = mxGPUCreateFromMxArray(prhs[1]);
    double const *opt;
    opt = mxGetPr(prhs[2]);
    P = opt[0];
    W = opt[1];
    if(mxGPUGetClassID(group) != mxDOUBLE_CLASS || mxGPUGetClassID(index) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    d_group = (const double *)(mxGPUGetDataReadOnly(group));
    d_index = (const double *)(mxGPUGetDataReadOnly(index));
    
    
    /* get dimensions */
    const mwSize *groupdim = mxGPUGetDimensions(group);
    const mwSize *indexdim = mxGPUGetDimensions(index);
    M = groupdim[0];
    N = groupdim[1];
    G = indexdim[2];
    mwSize imagedim[2];
    imagedim[0] = M;
    imagedim[1] = N;
    mwSize storedim[3];
    storedim[0] = M;
    storedim[1] = N;
    storedim[2] = G;
    double *d_image;
    double *d_overlap;
    double *d_storeIm;
    double *d_storeIn;
    image = mxGPUCreateGPUArray(2, imagedim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    overlap = mxGPUCreateGPUArray(2, imagedim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    storeIm = mxGPUCreateGPUArray(3, storedim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    storeIn = mxGPUCreateGPUArray(3, storedim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    d_image = (double *)(mxGPUGetData(image));
    d_overlap = (double *)(mxGPUGetData(overlap));
    d_storeIm = (double *)(mxGPUGetData(storeIm));
    d_storeIn = (double *)(mxGPUGetData(storeIn));
    /**/
    blocksPerGrid = (M*N*G + threadsPerBlock - 1) / threadsPerBlock;
    nnagg_ker<<<blocksPerGrid, threadsPerBlock>>>(d_group, d_index, d_storeIm, d_storeIn, M, N, G, P, W);
    blocksPerGrid = (M*N + threadsPerBlock - 1) / threadsPerBlock;
    nnsum_ker<<<blocksPerGrid, threadsPerBlock>>>(d_storeIm, d_storeIn, d_image, d_overlap, M, N, G);
    
    plhs[0] = mxGPUCreateMxArrayOnGPU(image);
    plhs[1] = mxGPUCreateMxArrayOnGPU(overlap);
    
    mxGPUDestroyGPUArray(group);
    mxGPUDestroyGPUArray(index);
    mxGPUDestroyGPUArray(storeIm);
    mxGPUDestroyGPUArray(storeIn);
    mxGPUDestroyGPUArray(image);
    mxGPUDestroyGPUArray(overlap);
}