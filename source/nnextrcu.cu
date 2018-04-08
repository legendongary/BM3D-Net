#include "mex.h"
#include "gpu/mxGPUArray.h"

#define IDX2(X, n1, n2, i1, i2) (X[(i2)*(n1) + (i1)])
#define IDX3(X, n1, n2, n3, i1, i2, i3) (X[(i3)*((n1)*(n2)) + (i2)*(n1) + (i1)])
#define IDX4(X, n1, n2, n3, n4, i1, i2, i3, i4) (X[(i4)*((n1)*(n2)*(n3)) + (i3)*((n1)*(n2)) + (i2)*(n1) + (i1)])

/* im2col: extract columns from padded image */
void __global__ im2col_ker(const double *Image, double *Colum, int psx, int psy, int isx, int isy)
{
    // psx: patch size x
    // psy: patch size y
    // isx: image size x
    // isy: image size y
    int M = psx * psy;
    int N = isx * isy;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(; n<N; n += total_threads)
    {
        int idx = n % isx;
        int idy = (n-idx) / isx;
        for(int i=0; i<psx; i++)
        {
            for(int j=0; j<psy; j++)
            {
                Colum[n*M + j*psx + i] = Image[(idy+j)*(isx+psx-1) + (idx+i)];
            }
        }
    }
}

/* nnextr: extract similar patch groups from columns according to index */
void __global__ nnextr_ker(const double *Patch, const double *Index, double *const Group, int psx, int psy, int isx, int isy, int grp)
{
    // psx: patch size x
    // psy: patch size y
    // isx: image size x
    // isy: image size y
    // grp: group number
    int M = psx * psy;
    int N = isx * isy;
    int G = grp;
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for(; n<N; n += total_threads)
    {
        int idx = n % isx;
        int idy = (n-idx) / isx;
        for(int i=0; i<G; i++)
        {
            int idk = IDX4(Index, isx, isy, grp, 2, idx, idy, i, 0) - 1;
            int idl = IDX4(Index, isx, isy, grp, 2, idx, idy, i, 1) - 1;
            for(int j=0; j<M; j++)
            {
                //Group[(i*M+j)*isx*isy + idy*isx + idx] = Patch[(idl*isx + idk)*M + j];
                IDX3(Group, isx, isy, M*G, idx, idy, i*M+j) = IDX2(Patch, M, N, j, idl*isx+idk);
            }
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *Image;
    mxGPUArray const *Index;
    mxGPUArray *Colum;
    mxGPUArray *Group;
    double const *d_Image;
    double const *d_Index;
    
    double const *Patch;
    double *d_Colum;
    double *d_Group;
    int M, N, G, isx, isy, psx, psy, grp;
    int i, j, k, l, m, n;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 256;
    int blocksPerGrid;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=3) || !(mxIsGPUArray(prhs[0])))
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    Image = mxGPUCreateFromMxArray(prhs[0]);
    Index = mxGPUCreateFromMxArray(prhs[1]);
    Patch = mxGetPr(prhs[2]);
    
    /* Verify that input are really DOUBLE data type. */
    if(mxGPUGetClassID(Image) != mxDOUBLE_CLASS || mxGPUGetClassID(Index) != mxDOUBLE_CLASS)
    {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    d_Image = (const double *)(mxGPUGetDataReadOnly(Image));
    d_Index = (const double *)(mxGPUGetDataReadOnly(Index));
    psx = Patch[0];
    psy = Patch[1];
    M = psx * psy;
    
    /* Get all kinds of dimensions. */    
    const mwSize *Iddim = mxGPUGetDimensions(Index);
    isx = Iddim[0];
    isy = Iddim[1];
    grp = Iddim[2];
    N = isx * isy;
    G = grp;
    mwSize Codim[2];
    Codim[0] = M;
    Codim[1] = N;
    mwSize Grdim[3];
    Grdim[0] = isx;
    Grdim[1] = isy;
    Grdim[2] = M * G;
    
    /* */
    Colum = mxGPUCreateGPUArray(2, Codim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    d_Colum = (double *)(mxGPUGetData(Colum));
    Group = mxGPUCreateGPUArray(3, Grdim, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    d_Group = (double *)(mxGPUGetData(Group));
    
    /**/
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    im2col_ker<<<blocksPerGrid, threadsPerBlock>>>(d_Image, d_Colum, psx, psy, isx, isy);
    nnextr_ker<<<blocksPerGrid, threadsPerBlock>>>(d_Colum, d_Index, d_Group, psx, psy, isx, isy, grp);
    plhs[0] = mxGPUCreateMxArrayOnGPU(Group);
    
    /**/
    mxGPUDestroyGPUArray(Image);
    mxGPUDestroyGPUArray(Index);
    mxGPUDestroyGPUArray(Colum);
    mxGPUDestroyGPUArray(Group);
}











