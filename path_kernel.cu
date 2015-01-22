//#include "path_kernel.h"
/*#include <gqd.cu>
 #include <gqd_type.h>
 #include "complex.h"
 #include <iomanip>
 #include <sys/time.h>*/

#include <gqd.cu>
#include <gqd_type.h>
#include "gpu_data.h"
#include "parameter.h"

#define maxrounds 128

#define max_array_size 2000
// Parameter for get_max_array_double
// Up to 4000

#define d  0
#define dd 1
#define qd 2

#if(precision == 0)
#define BS_QR 256
#include "path_kernel_d.cu"
#elif(precision == 1)
#define shmemsize    256
#define BS_QR 128
#include "path_kernel_dd.cu"
#else
#define shmemsize 128
#define BS_QR 64
#include "path_kernel_qd.cu"
#endif

__device__ inline int r_pos(int x, int y, int cols){
	return cols*(cols+1)/2 -1 - (y*(y+1)/2 -(x-y));
}

void cuda_set() {
	cudaSetDevice(0);
	if (cudaSuccess
			!= cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte))
		std::cout << "Error!\n" << std::endl;
	else {
		std::cout << "Success!\n" << std::endl;
	}
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}

__global__ void update_x(GT* x, GT* sol0, int dim)
{
	int BS = blockDim.x;
	int bidx = blockIdx.x*BS;
	int tidx = threadIdx.x;
	int idx = bidx + tidx;

	/*int sys_idx = blockIdx.z;
	 x_predictor += sys_idx*np_predictor*dim;
	 t_predictor += sys_idx*np_predictor;
	 x_new += sys_idx*dim;*/

	if(idx < dim) {
		x[idx] = x[idx] - sol0[idx];
	}
}

__global__ void small_backsubstitution(GT* R, GT*sol0, int dim)
{
	int j = threadIdx.x;
	__shared__ GT sol[shmemsize/2];
	__shared__ GT Rcl[shmemsize/2];
	int ind;
	GT update;

	update = R[j];
	for(int k=dim-1; k>=0; k--)  // compute k-th component of solution
	{
		if(j < k+1)
		{
			ind = (dim - k)*(dim + 3 + k)/2 + j;
			Rcl[j] = R[ind];
		}
		if(j == k) sol[j] = update/Rcl[j]; // all other threads wait
		__syncthreads();
		if(j < k) update = update - sol[k]*Rcl[j];// update
	}
	sol0[j] = sol[j];
}

__global__ void small_QR_normalize(GT* v, GT* R,
int dimR, int rows, int rowsLog2, int cols, int pivot ) {
	int b = blockIdx.x;
	int j = threadIdx.x;
	//int block = b+pivot;    // column for reduction w.r.t. pivot
	//int i = block*rows + j;// idx
	int L = pivot*rows + j;

	__shared__ GT piv[shmemsize];// contains pivot column
	__shared__ T prd[shmemsize];// for norm of the pivot

	piv[j] = v[L];
	prd[j] = piv[j].real*piv[j].real + piv[j].imag*piv[j].imag;
	__syncthreads();

	rowsLog2 -= 1;
	int half_size = 1 << (rowsLog2);// sum for the norm
	if(j + half_size < rows) {
		prd[j] = prd[j] + prd[j+half_size];
	}

	for(int k=0; k < rowsLog2; k++)
	{
		if(half_size > 16) {
			__syncthreads();
		}
		half_size /= 2;
		if(j < half_size) {
			prd[j] = prd[j] + prd[j+half_size];
		}
	}
	if(j == 0) prd[0] = sqrt(prd[0]);
	__syncthreads();

	piv[j] = piv[j]/prd[0];
	v[L] = piv[j];
	if(j == 0)
	{
		int indR = r_pos(pivot, pivot, cols); //(dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);
		R[indR].init_imag();
		R[indR].real = prd[0];
	}
}

__global__ void small_QR_reduce(GT* v, GT* R,
int dimR, int rows, int rowsLog2, int cols, int pivot )
{
	int b = blockIdx.x+1;
	int j = threadIdx.x;
	int block = b+pivot;    // column for reduction w.r.t. pivot
	int i = block*rows + j;// idx
	int L = pivot*rows + j;

	__shared__ GT piv[shmemsize/2 + 15];// contains pivot column
	__shared__ GT shv[2][shmemsize/2 + 15];// for the reduction

	int indR = r_pos(pivot, pivot+b, cols);// (dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);

	piv[j] = v[L];

	shv[0][j] = v[i];
	shv[1][j] = piv[j].adj()*shv[0][j];

	__syncthreads();

	rowsLog2 -= 1;
	int half_size = 1 << (rowsLog2);// sum for the norm
	if(j + half_size < rows) {
		shv[1][j] = shv[1][j] + shv[1][j+half_size];
	}

	for(int k=0; k < rowsLog2; k++)
	{
		if(half_size > 16) {
			__syncthreads();
		}
		half_size /= 2;
		if(j < half_size) {
			shv[1][j] = shv[1][j] + shv[1][j+half_size];
		}
	}
	__syncthreads();

	shv[0][j] = shv[0][j] - shv[1][0]*piv[j];
	v[i] = shv[0][j];
	if(j == 0) R[indR] = shv[1][0];
}

__global__ void large_QR_normalize_1(GT* v, GT* R, int rows, int rowsLog2, int cols,
int pivot, int rnd, int rndLog2, int BS, int BSLog2, T *pivnorm, int lastBSLog2, T* sums_global)
{
	int b = blockIdx.x;
	int j = threadIdx.x;
	int i = b*BS + j;

	__shared__ T shv[shmemsize/2];// for the reduction


	if(b == rnd-1){
		BSLog2 = lastBSLog2;
	}

	BSLog2 -= 1;

	if(i < rows) {
		GT tmp_piv = v[pivot*rows+i];
		shv[j] = tmp_piv.real*tmp_piv.real + tmp_piv.imag*tmp_piv.imag;
	}


	int half_size = 1 << (BSLog2);// sum for the norm

	if(half_size > 16) {
		__syncthreads();
	}

	if(i + half_size < rows && j < half_size) {
		shv[j] = shv[j] + shv[j+half_size];
	}

	for(int k=0; k < BSLog2; k++)
	{
		half_size /= 2;
		if(half_size > 16) {
			__syncthreads();
		}
		if(j < half_size) {
			shv[j] = shv[j] + shv[j+half_size];
		}
	}

	if(j == 0) sums_global[b] = shv[0];
}

__global__ void large_QR_normalize_2(GT* v, GT* R, int rows, int rowsLog2, int cols,
int pivot, int rnd, int rndLog2, int BS, int BSLog2, T *pivnorm, int lastBSLog2, T* sums_global)
{
	int b = blockIdx.x;
	int j = threadIdx.x;
	int i = b*BS + j;

	__shared__ T sums[maxrounds];// partial sums in rounds
	// maxrounds 32 reduce sync

	T newpivnorm;// norm of the pivot

	if(j < rnd){
		sums[j] = sums_global[j];
	}

	if(rndLog2 > 0){
		int powTwo = 1<<(rndLog2-1); // sum for the norm

		if(powTwo>16){
			__syncthreads();
		}

		if(j + powTwo < rnd) {
			sums[j] = sums[j] + sums[j+powTwo];
		}

		for(int k=0; k < rndLog2-1; k++)
		{
			if(powTwo>16){
				__syncthreads();
			}
			powTwo = powTwo/2;
			if(j < powTwo){
				sums[j] = sums[j] + sums[j+powTwo];
			}
		}
	}

	__syncthreads();

	newpivnorm = sqrt(sums[0]);

	if(i<rows)          // exclude extra threads in last round
	{
		v[pivot*rows+i] /= newpivnorm;
	}

	if(i == 0)
	{
		int indR = r_pos(pivot, pivot+b, cols);//(dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);
		R[indR].init(0.0,0.0);
		R[indR].real = newpivnorm;
	}
}

__global__ void large_QR_reduce(GT* v, GT* R, int cols, int rows, int rowsLog2, \
		int pivot, int rnd, int rndLog2, int BS, int BSLog2, T *pivnorm, int lastBSLog2, int piv_end = 0) {
	int b = blockIdx.x + 1 + piv_end;
	int j = threadIdx.x;
	int block = b+pivot; // column for reduction w.r.t. pivot
	int vBSind = 0;
	int powTwo;

	__shared__ GT piv[BS_QR];  // contains pivot column
	__shared__ GT shv[BS_QR];  // for the reduction
	__shared__ GT sums[maxrounds];   // partial sums in rounds

	vBSind = 0;
	for(int i=0; i<rnd; i++)// normalize and partial sums for inner product
	{
		if(vBSind+j<rows){
			piv[j] = v[pivot*rows+j+vBSind];
			shv[j] = v[block*rows+j+vBSind];
			shv[j] = piv[j].adj()*shv[j];
		}

		if(i==rnd-1){
			BSLog2 = lastBSLog2;
		}
		powTwo = 1<<(BSLog2-1); // sum for the norm

		if(powTwo > 16) {
			__syncthreads();
		}

		if(j+powTwo<BS && vBSind+j+powTwo<rows) {
			shv[j] = shv[j] + shv[j+powTwo];
		}

		for(int k=0; k<BSLog2-1; k++)
		{
			powTwo = powTwo/2;
			if(powTwo > 16) {
				__syncthreads();
			}
			if(j < powTwo){
				shv[j] = shv[j] + shv[j+powTwo];
			}
			//__syncthreads();
		}
		if(j == 0) sums[i] = shv[0];
		__syncthreads();  // avoid shv[0] is changed in next round
		vBSind = vBSind + BS;
	}

	if(rndLog2 > 0){
		powTwo = 1<<(rndLog2-1); // sum for the norm

		if(j + powTwo < rnd) {
			sums[j] = sums[j] + sums[j+powTwo];
		}

		for(int k=0; k < rndLog2-1; k++)
		{
			powTwo = powTwo/2;
			// Maxround < 32, so it is not necessary to sync.
			if(powTwo > 16) {
				__syncthreads();
			}
			if(j < powTwo){
				sums[j] = sums[j] + sums[j+powTwo];
			}
		}
	}

	__syncthreads();

	vBSind = 0;
	for(int i=0; i<rnd; i++)            // perform reduction
	{
		if(vBSind+j < rows)
		{
			piv[j] = v[pivot*rows+j+vBSind];
			shv[j] = v[block*rows+j+vBSind];
			shv[j] = shv[j] - sums[0]*piv[j];
			v[block*rows+j+vBSind] = shv[j];
		}
		vBSind = vBSind + BS;
	}
	if(j == 0){
		int indR = r_pos(pivot, pivot+b, cols);//(dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);
		R[indR] = sums[0];
	}
}

/*__global__ void large_QR_reduce(GT* v, GT* R, int cols, int rows, int rowsLog2, \
		int pivot, int rnd, int rndLog2, int BS, int BSLog2, T *pivnorm, int lastBSLog2, int piv_end = 0) {
	int b = blockIdx.x + 1 + piv_end;
	int j = threadIdx.x;
	int block = b+pivot; // column for reduction w.r.t. pivot
	int vBSind = 0;
	int powTwo;

	__shared__ GT piv[BS_QR];  // contains pivot column
	__shared__ GT shv[BS_QR];  // for the reduction
	__shared__ GT sum;

	if(j==0){
		sum = GT(0.0,0.0);
	}

	vBSind = 0;
	for(int i=0; i<rnd; i++)// normalize and partial sums for inner product
	{
		if(vBSind+j<rows){
			piv[j] = v[pivot*rows+j+vBSind];
			shv[j] = v[block*rows+j+vBSind];
			shv[j] = piv[j].adj()*shv[j];
		}

		if(i==rnd-1){
			BSLog2 = lastBSLog2;
		}
		powTwo = 1<<(BSLog2-1); // sum for the norm

		if(powTwo > 16) {
			__syncthreads();
		}

		if(j+powTwo<BS && vBSind+j+powTwo<rows) {
			shv[j] = shv[j] + shv[j+powTwo];
		}

		for(int k=0; k<BSLog2-1; k++)
		{
			powTwo = powTwo/2;
			if(powTwo > 16) {
				__syncthreads();
			}
			if(j < powTwo){
				shv[j] = shv[j] + shv[j+powTwo];
			}
			//__syncthreads();
		}
		if(j == 0) sum = sum + shv[0];
		__syncthreads();  // avoid shv[0] is changed in next round
		vBSind = vBSind + BS;
	}

	vBSind = 0;
	for(int i=0; i<rnd; i++)            // perform reduction
	{
		if(vBSind+j < rows)
		{
			piv[j] = v[pivot*rows+j+vBSind];
			shv[j] = v[block*rows+j+vBSind];
			shv[j] = shv[j] - sum*piv[j];
			v[block*rows+j+vBSind] = shv[j];
		}
		vBSind = vBSind + BS;
	}
	if(j == 0){
		int indR = r_pos(pivot, pivot+b, cols);//(dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);
		R[indR] = sum;
	}
}*/

__global__ void large_backsubstitution_1(GT* R, GT* x, int dim, int rnd, int pivot, int BS0, int BS)
{
	int j = threadIdx.x;
	__shared__ GT sol[BS_QR];
	__shared__ GT Rcl[BS_QR];
	int ind;
	GT update;
	int offset = pivot*BS;

    update = R[offset+j];
	//__syncthreads();
	for(int k=BS0-1; k>=0; k--)  // compute k-th component of solution
	{
		if(j < k+1)
		{
			ind = (dim - k - offset)*(dim + 3 + k + offset)/2 + j;
			Rcl[j] = R[ind+offset];
		}
		if(j == k) sol[j] = update/Rcl[j]; // all other threads wait
		__syncthreads();
		if(j < k) update = update - sol[k]*Rcl[j];// update
		//__syncthreads();
	}
	x[offset+j] = sol[j];
}

//large_backsubstitution_2<<<rf-1,BS>>>(R,sol,cols-1,rf,rf-1,n_cols_rest,BS);
__global__ void large_backsubstitution_2(GT* R, GT* x, int dim, int rnd, int pivot, int BS0, int BS)
{
	int j = threadIdx.x;
	int b = blockIdx.x+1;
	__shared__ GT sol[BS_QR];
	__shared__ GT Rcl[BS_QR];
	int ind;
	GT update;
	int offset = pivot*BS;//+BS0;

	if(j < BS0){
		sol[j] = x[offset+j];
	}

	int block_offset = b*BS;
	update = R[offset-block_offset+j];

	for(int k=BS0-1; k>=0; k--)  // continue updates
	{
		ind = (dim - k - offset)*(dim + 3 + k + offset)/2 + j;
		Rcl[j] = R[ind+offset-block_offset];
		update = update - sol[k]*Rcl[j];  // update
	}
	R[offset-block_offset+j] = update;
}

__global__ void large_backsubstitution(GT* R, GT* x, int dim, int rnd, int pivot, int BS )
{
	int j = threadIdx.x;
	int b = blockIdx.x;
	__shared__ GT sol[BS_QR];
	__shared__ GT Rcl[BS_QR];
	int ind;
	GT update;
	int offset = pivot*BS;

	if(pivot == rnd-1)
		update = R[offset+j];
	else
		update = x[offset+j];
	__syncthreads();
	for(int k=BS-1; k>=0; k--)  // compute k-th component of solution
	{
		if(j < k+1)
		{
			ind = (dim - k - offset)*(dim + 3 + k + offset)/2 + j;
			Rcl[j] = R[ind+offset];
		}
		if(j == k) sol[j] = update/Rcl[j]; // all other threads wait
		__syncthreads();
		if(j < k) update = update - sol[k]*Rcl[j];// update
		__syncthreads();
	}
	if(b == 0) x[offset+j] = sol[j];
	if(b != 0)
	{
		int block_offset = b*BS;
		if(pivot == rnd-1)
		update = R[offset-block_offset+j];
		else
		update = x[offset-block_offset+j];
		for(int k=BS-1; k>=0; k--)  // continue updates
		{
			ind = (dim - k - offset)*(dim + 3 + k + offset)/2 + j;
			__syncthreads();
			Rcl[j] = R[ind+offset-block_offset];
			__syncthreads();
			update = update - sol[k]*Rcl[j];  // update
		}
		__syncthreads();
		x[offset-block_offset+j] = update;
	}
}

// Mon block evalutation and differentiation on GPU
//template <unsigned int n_th>
__global__ void predict_newton_kernel(GT* x_array, GT* t_array, GT* x_new, GT* t_new, int n_predictor, int dim, int x_t_idx, int n_sys) {
	__shared__ GT div_diff_sh[shmemsize];
	__shared__ GT t_predictor[8]; // XXX n_predict
	int BS = blockDim.x;
	int bidx = blockIdx.x*BS;
	int tidx = threadIdx.x;
	int idx = bidx + tidx;

	/*int sys_idx = blockIdx.z;

	 x_predictor += sys_idx*n_predictor*dim;
	 t_predictor += sys_idx*n_predictor;
	 x_new += sys_idx*dim;*/

	// load t value
	if(tidx<n_predictor) {
		// XXXXXX Remove %
		t_predictor[tidx] = t_array[(tidx+x_t_idx+1)%(n_predictor+1)];
	}

	if(idx < dim) {
		GT* div_diff = div_diff_sh;
		//GT* div_diff = div_diff_sh + tidx; // XXX it can remove idx, not sure which one is better

		// Copy initial X value to divide difference
		int div_idx = 0;

		for(int np_idx = x_t_idx+1; np_idx < n_predictor+1; np_idx++) {
			div_diff[div_idx*BS + tidx]=x_array[np_idx*dim + idx];
			div_idx++;
		}

		for(int np_idx = 0; np_idx < x_t_idx; np_idx++) {
			div_diff[div_idx*BS + tidx]=x_array[np_idx*dim + idx];
			div_idx++;
		}

		// Compute divide difference
		for(int i = 1; i < n_predictor; i++) {
			for(int j = n_predictor-1; j >= i; j--) {
				div_diff[j*BS + tidx] = (div_diff[j*BS + tidx] - div_diff[(j-1)*BS + tidx])/(t_predictor[j]-t_predictor[j-i]);
			}
		}

		// Compute predict point
		GT x_tmp(0.0,0.0);
		for(int i=n_predictor-1; i > 0; i--) {
			x_tmp = (x_tmp + div_diff[i*BS + tidx]) * (*t_new - t_predictor[i-1]);
		}

		// Put X back
		x_new[idx] = x_tmp + div_diff[tidx];
	}
}

// Mon block evalutation and differentiation on GPU
//template <unsigned int n_th>
__global__ void eval_coef_kernel(GT* workspace_coef, const GT* coef_orig, int n_coef, GT* t, GT* one_minor_t, int n_sys) {
	//__shared__ GT div_diff_sh[shmemsize];
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x;
	int tidx = threadIdx.x;
	int idx = bidx + tidx;

	/*int sys_idx = blockIdx.z;
	 x_predictor += sys_idx*np_predictor*dim;
	 t_predictor += sys_idx*np_predictor;
	 x_new += sys_idx*dim;*/

	if(idx < n_coef) {
		//workspace_coef[idx] = coef_orig[idx];
		// XXX align coef later (*t)*coef_orig[idx] + (*one_minor_t)*coef_orig[idx+n_coef]
		workspace_coef[idx] = (*t)*coef_orig[2*idx] + (*one_minor_t)*coef_orig[2*idx+1];
	}
}

// Mon evalutaion and differentiation on GPU
__global__ void eval_sum_kernel(GT* workspace_matrix, GT* workspace_sum, int* sum_pos, int* sum_pos_start, int n_sum, int n_sys) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int sys_idx = blockIdx.z;
	//GT* x_d_tmp = x_d + sys_idx*dim;
	//GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

	//int tidx = threadIdx.x;
	if(idx < n_sum) {
		int* pos = sum_pos + sum_pos_start[idx];
		int n_var = *pos++;

		GT tmp = workspace_sum[*pos++];

		for(int i=1; i<n_var; i++) {
			tmp += workspace_sum[*pos++];
		}

		workspace_matrix[*pos] = tmp;
	}
}

// Sum block level 0
__global__ void eval_sum_block_0(GT* r_matrix_d, GT* workspace_d, int* sum_array_d, int* sum_start_d, int n_sum, int n_sys){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //int sys_idx = blockIdx.z;
    //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;
    //GT* r_matrix_d_tmp = r_matrix_d + sys_idx*r_matrix_size_int;

    if(idx < n_sum){
        int start_pos = sum_start_d[idx];
        int* sum_array = sum_array_d + start_pos;
        r_matrix_d[sum_array[2]] = workspace_d[sum_array[1]];
    }
}

// Sum block, mulitithread gpu sum, sum-2
__global__ void eval_sum_block_2(GT* r_matrix_d, GT* workspace_d, int* sum_array_d, int* sum_start_d, int n_sum, int n_sys){
    __shared__ GT x_sh[16];
    int tidx = threadIdx.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x;
    int idx = bidx + tidx;
    //int sys_idx = blockIdx.z;
    //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;
    //GT* r_matrix_d_tmp = r_matrix_d + sys_idx*r_matrix_size_int;

    if(idx < n_sum*2){
        int pidx = idx&1;
        int midx = tidx/2;
        int midx_global = idx/2;
        int start_pos = sum_start_d[midx_global];
        int* sum_array = sum_array_d + start_pos;
        int n_terms = *sum_array++;
        /*if(pidx == 0){
        	output= *sum_array_tmp;
        }
        sum_array_tmp++;*/
        GT tmp = workspace_d[sum_array[pidx]];
        int n_terms2 = n_terms/2*2;
        for(int i=2; i<n_terms2; i+=2){
        	tmp += workspace_d[sum_array[i+pidx]];
        }
        if(pidx == 1){
            if( n_terms2 < n_terms){
            	tmp += workspace_d[sum_array[n_terms2]];
            }
            x_sh[midx] = tmp;
        }
	__syncthreads();
        if(pidx == 0){
        	int output = sum_array[n_terms];
        	r_matrix_d[output] = tmp + x_sh[midx];
        }
    }
}


// Sum block, mulitithread gpu sum, sum-2, sum-4, sum-8
template <unsigned int n_th>
__global__ void eval_sum_block_n(GT* r_matrix_d, GT* workspace_d, int* sum_array_d, int* sum_start_d, int n_sum, int n_sys){
    __shared__ GT x_sh[32];
    int tidx = threadIdx.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x;
    int idx = bidx + tidx;

    if(idx < n_sum*n_th){
        //int sys_idx = blockIdx.z;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;
        //GT* r_matrix_d_tmp = r_matrix_d + sys_idx*r_matrix_size_int;

        int pidx = idx&(n_th-1);
        int midx = tidx/n_th;
        int midx_global = idx/n_th;
        int start_pos = sum_start_d[midx_global];
        int* sum_array_tmp = sum_array_d + start_pos;
        int n_terms = *sum_array_tmp++;

        GT tmp = workspace_d[sum_array_tmp[pidx]];
        int n_terms2 = n_terms/n_th*n_th;
        for(int i=n_th; i<n_terms2; i+=n_th){
        	tmp += workspace_d[sum_array_tmp[i+pidx]];
        }
        int pidx_last = n_terms2 + pidx;
        if( pidx_last < n_terms){
        	tmp += workspace_d[sum_array_tmp[pidx_last]];
        }
        x_sh[tidx] = tmp;

        GT* x_start = x_sh + midx*n_th;

        if(n_th > 32){
        	__syncthreads();
            if(pidx <  32){  x_start[pidx] = x_start[pidx] + x_start[pidx+ 32];}
        }


        if(n_th > 16){
            if(pidx <  16){  x_start[pidx] = x_start[pidx] + x_start[pidx+ 16];}
        }

        if(n_th > 8){
            if(pidx <  8){  x_start[pidx] = x_start[pidx] + x_start[pidx+ 8];}
        }

        if(n_th > 4){
            if(pidx <  4){  x_start[pidx] = x_start[pidx] + x_start[pidx+ 4];}
        }

        if(n_th > 2){
            if(pidx <  2){ x_start[pidx] = x_start[pidx] + x_start[pidx+ 2];}
        }

        if(n_th > 1){
            if(pidx <  1){ x_start[pidx] = x_start[pidx] + x_start[pidx+ 1];}
        }

        if(pidx == 0){
        	int output = sum_array_tmp[n_terms];
        	r_matrix_d[output] = x_start[0];
        	//r_matrix_d[output] = tmp;
        }
    }
    //r_matrix_d[0] = GT(n_sum, -2.0);
}

void eval_sum(GPUWorkspace& workspace, const GPUInst& inst, int n_sys){
	int sum_method = 2;
	if(sum_method == 0){
		eval_sum_kernel<<<inst.sum_grid, inst.sum_BS>>>(workspace.matrix,
				workspace.sum, inst.sum_pos, inst.sum_pos_start, inst.n_sum, n_sys);
	}
	else if(sum_method == 1){
		int BS = 32;
		int* sum_start_tmp = inst.sum_pos_start;
		int NB0 = (inst.n_sum_level[0]-1)/BS +1;
		dim3 nNB0(NB0,1,n_sys);
		//cout << "level "<< 0 << "NB0 = " << NB0 << endl;
		eval_sum_block_0<<<nNB0, BS>>>(workspace.matrix,
				workspace.sum, inst.sum_pos, inst.sum_pos_start, inst.n_sum_level[0], n_sys);

		sum_start_tmp += inst.n_sum_level[0];
		int n_sum_new = inst.n_sum_level_rest[0];
		int NBS = (n_sum_new-1)/16 +1;
		dim3 nNB(NBS,1,n_sys);
		//cout << "NBS = " << NBS << endl;
		eval_sum_block_2<<<nNB, BS>>>(workspace.matrix,
				workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
	}
	else{
		int max_level = 2;
        // d pieri 103 4
        // dd pieri 103 4
        // qd pieri 103 4
        //
        // d cyclic 352 2
        // dd cyclic 352 2
        // qd cyclic 352 2
        // qd cyclic 128 2
        // dd cyclic 128 2
        // d cyclic 128 2

		int last_level = min(max_level, inst.n_sum_levels);

		int* sum_start_tmp = inst.sum_pos_start;


		for(int i=0; i<last_level+1; i++){
			if(inst.n_sum_levels <= i){
				break;
			}

			int n_sum_new;
			dim3 sum_grid;
			if(i != last_level){
				n_sum_new = inst.n_sum_level[i];
				sum_grid = inst.sum_level_grid[i];
			}
			else{
				if(i > 0 ){
				    n_sum_new = inst.n_sum_level_rest[i-1];// inst.n_sum - inst.n_sum0;
				}
				else{
				    n_sum_new = inst.n_sum;// inst.n_sum - inst.n_sum0;
				}
				sum_grid = inst.sum_level_grid_rest[i];
			}

			if(i== 0 && n_sum_new > 0){
                if(max_level > 0){
				    eval_sum_block_0<<<sum_grid, inst.sum_BS>>>(workspace.matrix,
						    workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
                }
                else{
		            eval_sum_kernel<<<inst.sum_grid, inst.sum_BS>>>(workspace.matrix,
				           workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
                }
			}

			if(i== 1 && n_sum_new > 0){
				eval_sum_block_2<<<sum_grid, inst.sum_BS>>>(workspace.matrix,
						workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
			}

			if(i == 2 && n_sum_new > 0){
				eval_sum_block_n<4><<<sum_grid, inst.sum_BS>>>(workspace.matrix,\
						workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
			}

			if(i == 3 && n_sum_new > 0){
				eval_sum_block_n<8><<<sum_grid, inst.sum_BS>>>(workspace.matrix,\
						workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
			}

			if(i == 4 && n_sum_new > 0){
				eval_sum_block_n<16><<<sum_grid, inst.sum_BS>>>(workspace.matrix,\
						workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
			}

			if(i == 5 && n_sum_new > 0){
				eval_sum_block_n<32><<<sum_grid, inst.sum_BS>>>(workspace.matrix,\
						workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
			}

			/*if(i == 6 && n_sum_new > 0){
				eval_sum_block_n<64><<<sum_grid, inst.sum_BS>>>(workspace.matrix,\
						workspace.sum, inst.sum_pos, sum_start_tmp, n_sum_new, n_sys);
			}*/
			sum_start_tmp += n_sum_new;
		}
	}
}

void eval(GPUWorkspace& workspace, const GPUInst& inst, int n_sys = 1) {

	eval_coef_kernel<<<inst.coef_grid, inst.coef_BS>>>(workspace.coef,
			inst.coef, inst.n_coef, workspace.t, workspace.one_minor_t, n_sys);

	eval_mon(workspace, inst, n_sys);

	eval_sum(workspace, inst, n_sys);

}

void small_MGS_with_delta(GT* V, GT* R, GT* sol, int rows, int cols, double* max_delta_x) {
	int BS = rows; // XXX Temperary solution
	//int rows = dim;
	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	int dimR = cols*(cols+1)/2;
	int dimLog2 = log2ceil(cols-1);// ceil for sum reduction

	for(int piv=0; piv<cols-1; piv++) {
		small_QR_normalize<<<1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
		small_QR_reduce<<<cols-piv-1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
	}
	small_backsubstitution_with_delta<<<1,cols-1>>>(R,sol,cols-1, dimLog2, max_delta_x);
}

void small_MGS(GT* V, GT* R, GT* sol, int rows, int cols) {
	int BS = rows; // XXX Temperary solution
	//int rows = dim;
	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	int dimR = cols*(cols+1)/2;
	//int cols = dim + 1;

	/*std::cout << "rows = " << rows
	<< " cols = " << cols
	<< " rowsLog2 = " << rowsLog2 << std::endl;*/

	for(int piv=0; piv<cols-1; piv++) {
		small_QR_normalize<<<1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
		small_QR_reduce<<<cols-piv-1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
	}
	small_backsubstitution<<<1,cols-1>>>(R,sol,cols-1);
}

void large_MGS(GT* V, GT* R, GT* sol, int rows, int cols, double* max_delta_x=NULL) {
	int BS = min(BS_QR,rows);
	//int BS = 32;

	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	//int dimR = cols*(cols+1)/2;

	T* pivnrm;// norm of the pivot column
	cudaMalloc((void**)&pivnrm,sizeof(T));

	T* sums_global;// norm of the pivot column
	cudaMalloc((void**)&sums_global,maxrounds*sizeof(T));

	int rf = ceil(((double) rows)/BS);
	int rfLog2 = log2ceil(rf);
	int BSLog2 = log2ceil(BS);
	int lastBSLog2 = log2ceil(rows-BS*(rf-1));

	/*std::cout << "BS     = " << BS << std::endl;
	std::cout << "rf     = " << rf << std::endl;
	std::cout << "rfLog2 = " << rfLog2 << std::endl;
	std::cout << "BSLog2 = " << BSLog2 << std::endl;
	std::cout << "lastBSLog2 = " << lastBSLog2 << std::endl;*/

	for(int piv=0; piv<cols-1; piv++) {
		large_QR_normalize_1<<<rf,BS>>>
		(V,R,rows,rowsLog2,cols,piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2,sums_global);
		large_QR_normalize_2<<<rf,BS>>>
		(V,R,rows,rowsLog2,cols,piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2,sums_global);
		// XXX BS should be greater than maxround
		large_QR_reduce<<<cols-piv-1,BS>>>
		(V,R,cols,rows,rowsLog2, piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2);
	}

	BS = BS_QR;

	/*rf = ceil(((double) (cols-1))/BS);
	for(int piv=rf-1; piv>=0; piv--) {
		large_backsubstitution<<<piv+1,BS>>>(R,sol,cols-1,rf,piv,BS);
	}*/

	rf = ceil(((double) (cols-1))/BS);
	//std::cout << "rf = " << rf << std::endl;
	int BS_col = cols-1 - BS*(rf-1);

	//std::cout << "BS_col = " << BS_col << std::endl;

	for(int piv=rf-1; piv>=0; piv--) {
		//std::cout<< "piv = " << piv << std::endl;
		if(piv==rf-2)	BS_col = BS;
		large_backsubstitution_1<<<1,BS_col>>>(R,sol,cols-1,rf,piv,BS_col,BS);
		if(piv==0)		break;
		large_backsubstitution_2<<<piv,BS>>>(R,sol,cols-1,rf,piv,BS_col,BS);
	}

	int dimLog2 = log2ceil(cols-1); // ceil for sum reduction
	if(max_delta_x != NULL) {
		get_max_array_double<<<1,cols-1>>>(sol, cols-1, dimLog2, max_delta_x);
	}
}

/*void large_MGS1(GT* V, GT* R, GT* sol, int rows, int cols, double* max_delta_x=NULL) {
	int BS = 8;

	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	int dimR = cols*(cols+1)/2;

	T* pivnrm;// norm of the pivot column
	cudaMalloc((void**)&pivnrm,sizeof(T));

	T* sums_global;// norm of the pivot column
	cudaMalloc((void**)&sums_global,maxrounds*sizeof(T));

	int rf = ceil(((double) rows)/BS);
	int rfLog2 = log2ceil(rf);
	int BSLog2 = log2ceil(BS);
	int lastBSLog2 = log2ceil(rows-BS*(rf-1));

	std::cout << "rows   = " << rows << std::endl;
	std::cout << "cols   = " << cols << std::endl;
	std::cout << "BS     = " << BS << std::endl;
	std::cout << "rf     = " << rf << std::endl;
	std::cout << "rfLog2 = " << rfLog2 << std::endl;
	std::cout << "BSLog2 = " << BSLog2 << std::endl;
	std::cout << "lastBSLog2 = " << lastBSLog2 << std::endl;

	int row_block = (rows-1)/matrix_block_row+1;
	int col_block = (cols-2)/matrix_block_pivot_col+1; // last column doesn't count

	std::cout << "row_block = " << row_block << std::endl;
	std::cout << "col_block = " << col_block << std::endl;

	for(int col_block_idx=0; col_block_idx<col_block; col_block_idx++){
		int piv_start = col_block_idx*matrix_block;
		int piv_end;
		if(col_block_idx != col_block-1){
			 piv_end = (col_block_idx+1)*matrix_block;
		}
		else{
			 piv_end = cols-1;
		}

		for(int piv=piv_start; piv<piv_end; piv++) {
			large_QR_normalize_1<<<rf,BS>>>
			(V,R,rows,rowsLog2,cols,piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2,sums_global);
			large_QR_normalize_2<<<rf,BS>>>
			(V,R,rows,rowsLog2,cols,piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2,sums_global);
			// XXX BS should be greater than maxround
			large_QR_reduce<<<piv_end-piv,BS>>>
			//large_QR_reduce<<<cols-1-piv,BS>>>
			(V,R,cols,rows,rowsLog2, piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2);
		}
		if(col_block_idx != col_block-1){
			std::cout << "piv_start = " << piv_start << ", piv_end = " << piv_end << std::endl;
			for(int reduce_idx = col_block_idx+1; reduce_idx < col_block; reduce_idx++){
				int reduce_cols;
				if(reduce_idx != col_block-1){
					reduce_cols = matrix_block;
				}
				else{
					reduce_cols = cols-1-reduce_idx*matrix_block;
				}
				for(int piv=piv_start; piv<piv_end; piv++) {
					large_QR_reduce<<<reduce_cols,BS>>>
					(V,R,dimR,rows,rowsLog2, piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2, piv_end-piv+matrix_block_pivot_col*(reduce_idx-col_block_idx-1));
				}
			}
		}
	}

	BS = BS_QR;

	rf = ceil(((double) (cols-1))/BS);
	//std::cout << "rf = " << rf << std::endl;
	int BS_col = cols-1 - BS*(rf-1);

	//std::cout << "BS_col = " << BS_col << std::endl;

	for(int piv=rf-1; piv>=0; piv--) {
		//std::cout<< "piv = " << piv << std::endl;
		if(piv==rf-2)	BS_col = BS;
		large_backsubstitution_1<<<1,BS_col>>>(R,sol,cols-1,rf,piv,BS_col,BS);
		if(piv==0)		break;
		large_backsubstitution_2<<<piv,BS>>>(R,sol,cols-1,rf,piv,BS_col,BS);
	}

	int dimLog2 = log2ceil(cols-1); // ceil for sum reduction
	if(max_delta_x != NULL) {
		get_max_array_double<<<1,cols-1>>>(sol, cols-1, dimLog2, max_delta_x);
	}
}*/

__global__ void matrix_init(GT* V, int rows, int cols){
    int tidx = threadIdx.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x;
    int idx = tidx + bidx;
    if(idx<rows*cols){
    	V[idx] = GT(1,0);
    	//V[idx] = GT(idx,0);
    }
}
__global__ void matrix_init_zero(GT* V, int rows, int cols){
    int tidx = threadIdx.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x;
    int idx = tidx + bidx;
    if(idx<rows*cols){
    	V[idx] = GT(0,0);
    }
}

template <unsigned int row_block_size>
__global__ void QR_reduce_block_1_1(GT* V, GT* P, int row_block, int col_block, int rows, int cols, int piv_col_block)
/*
 * Constant: matrix_block_row, matrix_block_pivot_col, matrix_block_reduce_col
 * row_block_size = matrix_block_row
 */
{
	int b = gridDim.x*blockIdx.y+blockIdx.x;
	int tidx = threadIdx.x;

	__shared__ GT pivot_block[matrix_block_row];  // contains pivot column
	__shared__ GT product_array[matrix_block_row];  // contains pivot column

	int row_block_idx = b-b/row_block*row_block; // b-(col_block_idx-1)*row_block
	int reduce_start_col = b/row_block*matrix_block_reduce_col + piv_col_block + 1;

	GT* pivot_block_start = V + piv_col_block*rows + row_block_idx*row_block_size;
	GT* reduce_block_start = V + reduce_start_col*rows + row_block_idx*row_block_size;
	int kernel_rows = rows - row_block_idx*row_block_size;


	// load pivot block and reduce block
	if(tidx < kernel_rows){
		pivot_block[tidx] = pivot_block_start[tidx];
	}

	int reduce_block_size = min(cols-reduce_start_col, matrix_block_reduce_col);

	// Compute product
	for(int reduce_idx=0; reduce_idx<reduce_block_size; reduce_idx++){
		if(tidx < kernel_rows){
			product_array[tidx] = reduce_block_start[reduce_idx*rows+tidx];
			//product_array[tidx] =  pivot_block[piv_idx*row_block_size+tidx].adj()*reduce_block[tidx];
			product_array[tidx] =  pivot_block[tidx].adj_multiple(product_array[tidx]);
		}
		if(row_block_size > 64){
			__syncthreads();
			if(tidx < 64 && tidx+64 < kernel_rows){
				product_array[tidx] = product_array[tidx] + product_array[tidx+64];
			}
		}
		if(row_block_size > 32){
			__syncthreads();
			if(tidx < 32 && tidx+32 < kernel_rows){
				product_array[tidx] = product_array[tidx] + product_array[tidx+32];
			}
		}
		if(row_block_size > 16){
			if(tidx < 16 && tidx+16 < kernel_rows){
				product_array[tidx] = product_array[tidx] + product_array[tidx+16];
			}
		}
		if(row_block_size > 8){
			if(tidx < 8 && tidx+8 < kernel_rows){
				product_array[tidx] = product_array[tidx] + product_array[tidx+8];
			}
		}
		if(row_block_size > 4){
			if(tidx < 4 && tidx+4 < kernel_rows){
				product_array[tidx] = product_array[tidx] + product_array[tidx+4];
			}
		}
		if(row_block_size > 2){
			if(tidx < 2 && tidx+2 < kernel_rows){
				product_array[tidx] = product_array[tidx] + product_array[tidx+2];
			}
		}
		if(row_block_size > 1){
			if(tidx < 1 && tidx+1 < kernel_rows){
				product_array[0] = product_array[0] + product_array[1];
			}
		}
		if(tidx == 0){
			int col_idx = reduce_start_col+reduce_idx;
			P[row_block_idx*cols+col_idx] = product_array[0];
		}
		if(row_block_size > 32){
			__syncthreads();
        	}
	}
}



template <unsigned int row_block_size>
__global__ void QR_reduce_block_1(GT* V, GT* P, int row_block, int col_block, int rows, int cols, int piv_col_block)
/*
 * Constant: matrix_block_row, matrix_block_pivot_col, matrix_block_reduce_col
 * row_block_size = matrix_block_row
 */
{
	int b = gridDim.x*blockIdx.y+blockIdx.x;
	int tidx = threadIdx.x;

	__shared__ GT pivot_block[matrix_block_row*matrix_block_pivot_col];  // contains pivot column
	__shared__ GT reduce_block[matrix_block_row];  // contains pivot column
	__shared__ GT product_array[matrix_block_row];  // contains pivot column

	int row_block_idx = b-b/row_block*row_block; // b-(col_block_idx-1)*row_block
	int reduce_start_col = b/row_block*matrix_block_reduce_col + (piv_col_block + 1)*matrix_block_pivot_col;

	GT* pivot_block_start = V + piv_col_block*matrix_block_pivot_col*rows + row_block_idx*row_block_size;
	GT* reduce_block_start = V + reduce_start_col*rows + row_block_idx*row_block_size;
	int kernel_rows = rows - row_block_idx*row_block_size;


	// load pivot block and reduce block
	for(int i=0; i<matrix_block_pivot_col; i++){
		if(tidx < kernel_rows){
			pivot_block[i*row_block_size+tidx] = pivot_block_start[i*rows+tidx];
		}
	}

	int reduce_block_size = min(cols-reduce_start_col, matrix_block_reduce_col);

	// Compute product
	for(int reduce_idx=0; reduce_idx<reduce_block_size; reduce_idx++){
		if(tidx < kernel_rows){
			reduce_block[tidx] = reduce_block_start[reduce_idx*rows+tidx];
		}
		for(int piv_idx=0; piv_idx<matrix_block_pivot_col; piv_idx++){
			if(tidx<kernel_rows){
				//product_array[tidx] =  pivot_block[piv_idx*row_block_size+tidx].adj()*reduce_block[tidx];
				product_array[tidx] =  pivot_block[piv_idx*row_block_size+tidx].adj_multiple(reduce_block[tidx]);
			}
			if(row_block_size > 128){
				__syncthreads();
				if(tidx < 128 && tidx+128 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+128];
				}
			}
			if(row_block_size > 64){
				__syncthreads();
				if(tidx < 64 && tidx+64 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+64];
				}
			}
			if(row_block_size > 32){
				__syncthreads();
				if(tidx < 32 && tidx+32 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+32];
				}
			}
			if(row_block_size > 16){
				if(tidx < 16 && tidx+16 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+16];
				}
			}
			if(row_block_size > 8){
				if(tidx < 8 && tidx+8 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+8];
				}
			}
			if(row_block_size > 4){
				if(tidx < 4 && tidx+4 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+4];
				}
			}
			if(row_block_size > 2){
				if(tidx < 2 && tidx+2 < kernel_rows){
					product_array[tidx] = product_array[tidx] + product_array[tidx+2];
				}
			}
			if(row_block_size > 1){
				if(tidx < 1 && tidx+1 < kernel_rows){
					product_array[0] = product_array[0] + product_array[1];
				}
			}
			if(tidx == 0){
				int col_idx = reduce_start_col+reduce_idx;
				P[row_block_idx*cols*matrix_block_pivot_col+col_idx*matrix_block_pivot_col+piv_idx] = product_array[0];
			}
			if(row_block_size > 32){
				__syncthreads();
			}
		}
	}
}

//template <unsigned int n_th>
__global__ void QR_reduce_block_3_1(GT* V, GT* R, int row_block, int col_block, int rows, int cols, int piv_col_block)
/*
 * Constant: matrix_block_row, matrix_block_pivot_col, matrix_block_reduce_col
 */
{
	int b = gridDim.x*blockIdx.y+blockIdx.x;
	int tidx = threadIdx.x;

	__shared__ GT pivot_block[matrix_block_row*matrix_block_pivot_col];  // contains pivot column
	__shared__ GT reduce_block[matrix_block_row];  // contains reduce column
	//__shared__ GT norm_array[matrix_block_pivot_col];  // contains pivot column

	int col_block_idx = b/row_block + 1 + piv_col_block;
	int row_block_idx = b-b/row_block*row_block; // b-(col_block_idx-1)*row_block
	int reduce_start_col = b/row_block*matrix_block_reduce_col + (piv_col_block + 1);

	GT* pivot_block_start = V + piv_col_block*rows + row_block_idx*matrix_block_row;
	GT* reduce_block_start = V + reduce_start_col*rows + row_block_idx*matrix_block_row;
	int kernel_rows = rows - row_block_idx*matrix_block_row;

	int reduce_block_size = min(cols-reduce_start_col, matrix_block_reduce_col);

	if(tidx < kernel_rows){
		// load pivot block and reduce block
		pivot_block[tidx] = pivot_block_start[tidx];

		for(int reduce_idx=0; reduce_idx<reduce_block_size; reduce_idx++){
			reduce_block[tidx] = reduce_block_start[reduce_idx*rows+tidx];
			int start_r_pos = r_pos(piv_col_block*matrix_block_pivot_col,reduce_start_col+reduce_idx, cols);
			reduce_block_start[reduce_idx*rows+tidx] = reduce_block[tidx] - R[start_r_pos]*pivot_block[tidx];
		}
	}
}

//template <unsigned int n_th>
__global__ void QR_reduce_block_3(GT* V, GT* R, int row_block, int col_block, int rows, int cols, int piv_col_block)
/*
 * Constant: matrix_block_row, matrix_block_pivot_col, matrix_block_reduce_col
 */
{
	int b = gridDim.x*blockIdx.y+blockIdx.x;
	int tidx = threadIdx.x;

	__shared__ GT pivot_block[matrix_block_row*matrix_block_pivot_col];  // contains pivot column
	__shared__ GT reduce_block[matrix_block_row];  // contains reduce column
	__shared__ GT norm_array[matrix_block_pivot_col];  // contains pivot column

	int col_block_idx = b/row_block + 1 + piv_col_block;
	int row_block_idx = b-b/row_block*row_block; // b-(col_block_idx-1)*row_block
	int reduce_start_col = b/row_block*matrix_block_reduce_col + (piv_col_block + 1)*matrix_block_pivot_col;

	GT* pivot_block_start = V + piv_col_block*matrix_block_pivot_col*rows + row_block_idx*matrix_block_row;
	GT* reduce_block_start = V + reduce_start_col*rows + row_block_idx*matrix_block_row;
	int kernel_rows = rows - row_block_idx*matrix_block_row;

	int reduce_block_size = min(cols-reduce_start_col, matrix_block_reduce_col);

	if(tidx < kernel_rows){
		// load pivot block and reduce block
		for(int i=0; i<matrix_block_pivot_col; i++){
			pivot_block[i*matrix_block_row+tidx] = pivot_block_start[i*rows+tidx];
		}

		for(int reduce_idx=0; reduce_idx<reduce_block_size; reduce_idx++){
			reduce_block[tidx] = reduce_block_start[reduce_idx*rows+tidx];
			GT tmp = reduce_block[tidx];
			for(int piv_idx=0; piv_idx<matrix_block_pivot_col; piv_idx++){
				//]tmp = tmp - Norm[(b/row_block*matrix_block_reduce_col+reduce_idx)*matrix_block_pivot_col+piv_idx]*pivot_block[piv_idx*matrix_block_row+tidx];

				int start_r_pos = r_pos(piv_col_block*matrix_block_pivot_col+piv_idx,reduce_start_col+reduce_idx, cols);
				tmp = tmp - R[start_r_pos]*pivot_block[piv_idx*matrix_block_row+tidx];
			}
			reduce_block_start[reduce_idx*rows+tidx] = tmp;
		}
	}
}

__global__ void QR_reduce_block_2(GT* P, GT* R, int row_block, int P_rows, int n_sum, int col_block_size, int cols, int piv_block){
	int b = gridDim.x*blockIdx.y+blockIdx.x;
	int tidx = threadIdx.x;
	int idx = b*blockDim.x+tidx;

	/*int block = b+pivot; // column for reduction w.r.t. pivot
	int vBSind = 0;
	int powTwo;
	int indR = (dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);*/

	// Compute product
	if(idx<n_sum){
		GT sum = P[idx];
		for(int i=1; i< row_block; i++){
			sum = sum + P[idx+i*P_rows];
		}

		int col_idx = idx-idx/col_block_size*col_block_size;
		int r_x_idx = piv_block*col_block_size;
		int r_y_idx = r_x_idx + col_block_size + idx/col_block_size;
		//r_x_idx += col_idx;

		int start_r_pos = r_pos(r_x_idx,r_y_idx, cols);

		//Norm[idx] = sum;//GT(idx, 0);
		R[start_r_pos+col_idx] = sum;
	}
}

void large_reduce(GT* V, GT* R, GT* P, int rows, int cols, int pivot_block){
	int BS = matrix_block_pivot_col;

	//std::cout << "rows   = " << rows << std::endl;
	//std::cout << "cols   = " << cols << std::endl;
	//std::cout << "BS     = " << BS << std::endl;

	int row_block = (rows-1)/matrix_block_row+1;
	int col_block = (cols-1)/matrix_block_pivot_col+1; // last column doesn't count

	//std::cout << "row_block = " << row_block << std::endl;
	//std::cout << "col_block = " << col_block << std::endl;

	int NB = (rows*cols-1)/BS+1;
	//matrix_init<<<NB, BS>>>(V, rows, cols);

	//NB = (row_block*matrix_block_pivot_col*cols-1)/BS+1;
	//matrix_init_zero<<<NB, BS>>>(P, row_block, matrix_block_pivot_col*cols);

	//GT* P_host = (GT*)malloc(row_block*matrix_block_pivot_col*cols*sizeof(GT));
	//matrix_init<<<NB, BS>>>(P, col_block*row_block, cols-1);

	NB = row_block*((cols-(pivot_block+1)*matrix_block_pivot_col-1)/matrix_block_reduce_col+1);
	//std::cout << "NB = " << NB << std::endl;
	dim3 NB3 = get_grid(NB,1);
	if(matrix_block_pivot_col==1){
		QR_reduce_block_1_1<matrix_block_row><<<NB3,matrix_block_row>>>(V, P, row_block, col_block, rows, cols, pivot_block);
	}
	else{
		QR_reduce_block_1<matrix_block_row><<<NB3,matrix_block_row>>>(V, P, row_block, col_block, rows, cols, pivot_block);
	}

	/*cudaMemcpy(P_host, P, row_block*matrix_block_pivot_col*cols*sizeof(GT),
			cudaMemcpyDeviceToHost);

	std::cout << "------------- Matrix P -----------" << std::endl;
	for(int col_idx=0; col_idx<cols; col_idx++){
		for(int block_col_idx=0; block_col_idx<matrix_block_pivot_col; block_col_idx++){
			for(int row_block_idx=0; row_block_idx<row_block; row_block_idx++){
				GT tmp_matrix = P_host[row_block_idx*cols*matrix_block_pivot_col+col_idx*matrix_block_pivot_col+block_col_idx];
				std::cout << "col=" << col_idx \
						  << " col_block=" << block_col_idx \
						  << " row_block=" << row_block_idx \
						  << "   " << tmp_matrix.real << " + " << tmp_matrix.imag << std::endl;
			}
		}
	}*/

	NB = (matrix_block_pivot_col*cols-1)/BS+1;
	//GT* Norm;
	//cudaMalloc((void**)&Norm, matrix_block_pivot_col*cols*sizeof(GT));
	//matrix_init_zero<<<NB, BS>>>(Norm, matrix_block_pivot_col, cols);

	int BS_sum = 32;
	int P_rows = cols*matrix_block_pivot_col;
	int n_sum = (cols-(pivot_block+1)*matrix_block_pivot_col)*matrix_block_pivot_col;
	NB = (n_sum-1)/BS_sum+1;
	//std::cout << "n_sum = " << n_sum
	//		  << " start = " << (cols-1)*col_block << std::endl;
	NB3 = get_grid(NB,1);
	QR_reduce_block_2<<<NB3,BS_sum>>>(P+(pivot_block+1)*matrix_block_pivot_col*matrix_block_pivot_col, R, \
			                         row_block, P_rows, n_sum, matrix_block_pivot_col, cols, pivot_block);

	/*std::cout << "------------- Norm -----------" << std::endl;
	GT* Norm_host = (GT*)malloc(matrix_block_pivot_col*cols*sizeof(GT));

	cudaMemcpy(Norm_host, Norm, matrix_block_pivot_col*cols*sizeof(GT),
			cudaMemcpyDeviceToHost);

	for(int i=0; i <cols*matrix_block_pivot_col; i++){
		std::cout << i << " " << Norm_host[i].real << " + " << Norm_host[i].imag << std::endl;
	}*/

	NB = row_block*((cols-(pivot_block+1)*matrix_block_pivot_col-1)/matrix_block_reduce_col+1);
	//std::cout << "NB = " << NB << std::endl;

	NB3 = get_grid(NB,1);
	if(matrix_block_pivot_col==1){
		QR_reduce_block_3_1<<<NB3,matrix_block_row>>>(V, R, row_block, col_block, rows, cols, pivot_block);
	}
	else{
		QR_reduce_block_3<<<NB3,matrix_block_row>>>(V, R, row_block, col_block, rows, cols, pivot_block);
	}
}

void large_MGS1(GT* V, GT* R, GT* P, GT* sol, int rows, int cols, double* max_delta_x=NULL) {

	/*std::cout << "------------- Matrix P Seq-----------" << std::endl;
	GT* tmp_matrix = P_host;
	int tmp_idx = 0;
	for(int col_idx=0; col_idx<cols-1; col_idx++){
		for(int col_block_idx=0; col_block_idx<col_block; col_block_idx++){
			for(int row_block_idx=0; row_block_idx<row_block; row_block_idx++){
				std::cout << tmp_idx++ << " "<< (*tmp_matrix).real << " + " << (*tmp_matrix).imag << std::endl;
				tmp_matrix++;
			}
		}
	}*/


	int BS = min(BS_QR,rows);
	//int BS = 32;

	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	//int dimR = cols*(cols+1)/2;

	T* pivnrm;// norm of the pivot column
	cudaMalloc((void**)&pivnrm,sizeof(T));

	T* sums_global;// norm of the pivot column
	cudaMalloc((void**)&sums_global,maxrounds*sizeof(T));

	int rf = (rows-1)/BS+1;
	int rfLog2 = log2ceil(rf);
	int BSLog2 = log2ceil(BS);
	int lastBSLog2 = log2ceil(rows-BS*(rf-1));

	int col_block = (cols-1)/matrix_block_pivot_col + 1;

	int NB = (rows*cols-1)/BS+1;
	//matrix_init<<<NB, BS>>>(V, rows, cols);

	//std::cout << "rf = " << rf << std::endl;
	//std::cout << "col_block = " << col_block << std::endl;

	for(int col_block_idx=0; col_block_idx<col_block; col_block_idx++){
		int piv_start = col_block_idx*matrix_block_pivot_col;
		int piv_end;
		if(col_block_idx != col_block-1){
			 piv_end = (col_block_idx+1)*matrix_block_pivot_col;
		}
		else{
			 piv_end = cols-1;
		}

		//std::cout << "piv_start = " << piv_start << ", piv_end = " << piv_end << std::endl;
		for(int piv=piv_start; piv<piv_end; piv++) {
			large_QR_normalize_1<<<rf,BS>>>
			(V,R,rows,rowsLog2,cols,piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2,sums_global);
			large_QR_normalize_2<<<rf,BS>>>
			(V,R,rows,rowsLog2,cols,piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2,sums_global);
			// XXX BS should be greater than maxround
			large_QR_reduce<<<piv_end-piv-1,BS>>>
			//large_QR_reduce<<<cols-1-piv,BS>>>
			(V,R,cols,rows,rowsLog2, piv,rf,rfLog2,BS,BSLog2,pivnrm,lastBSLog2);
		}
		if(col_block_idx != col_block-1){
				large_reduce(V, R, P, rows, cols, col_block_idx);
		}
	}

	BS = BS_QR;

	rf = ceil(((double) (cols-1))/BS);
	//std::cout << "rf = " << rf << std::endl;
	int BS_col = cols-1 - BS*(rf-1);

	//std::cout << "BS_col = " << BS_col << std::endl;

	for(int piv=rf-1; piv>=0; piv--) {
		//std::cout<< "piv = " << piv << std::endl;
		if(piv==rf-2)	BS_col = BS;
		large_backsubstitution_1<<<1,BS_col>>>(R,sol,cols-1,rf,piv,BS_col,BS);
		if(piv==0)		break;
		large_backsubstitution_2<<<piv,BS>>>(R,sol,cols-1,rf,piv,BS_col,BS);
	}

	int dimLog2 = log2ceil(cols-1); // ceil for sum reduction
	if(max_delta_x != NULL) {
		get_max_array_double<<<1,cols-1>>>(sol, cols-1, dimLog2, max_delta_x);
	}
}

bool newton(GPUWorkspace& workspace, const GPUInst& inst, Parameter path_parameter) {
	bool success = 1;
	int rowsLog2 = log2ceil(inst.n_eq); // ceil for sum reduction

	double* max_delta_x_gpu;
	cudaMalloc((void **) &max_delta_x_gpu, sizeof(double));

	double* max_f_val_gpu;
	cudaMalloc((void **) &max_f_val_gpu, sizeof(double));

	double last_delta_x = 1E10;
	double last_f_val   = 1E10;

	for (int i = 0; i < path_parameter.max_it; i++) {
		cout << "  Iteration " << i << endl;

		double max_delta_x;
		double max_f_val;

		eval(workspace, inst);

		get_max_array_double<<<1, inst.n_eq>>>(workspace.f_val, inst.n_eq,
				rowsLog2, max_f_val_gpu);

		cudaMemcpy(&max_f_val, max_f_val_gpu, sizeof(double),
				cudaMemcpyDeviceToHost);

		std::cout << "       max_f_value  = " << max_f_val << std::endl;

		if (max_f_val > path_parameter.err_max_res) {
			success = 0;
			break;
		}

        if(max_f_val < path_parameter.err_min_round_off){
        	// last_delta_x might be problem for constant 0
        	if(last_delta_x == 1E10){
        		last_delta_x = 0;
        	}
        	break;
        }

		if (inst.dim <= BS_QR) {
			small_MGS_with_delta(workspace.matrix, workspace.R, workspace.sol,
					inst.n_eq, inst.dim + 1, max_delta_x_gpu);
		} else {
			large_MGS1(workspace.matrix, workspace.R, workspace.P, workspace.sol, inst.n_eq,\
					inst.dim + 1, max_delta_x_gpu);
			//large_MGS(workspace.V, workspace.R, workspace.sol, inst.n_eq, inst.dim+1);
		}

		cudaMemcpy(&max_delta_x, max_delta_x_gpu, sizeof(double),
				cudaMemcpyDeviceToHost);

		//gqd2qd(max_delta_x_host, &(max_delta_x));

		std::cout << "       max_delta_x  = " << max_delta_x << std::endl;

		if (max_delta_x < path_parameter.err_min_round_off) {
			last_delta_x = max_delta_x;
			break;
		}

		if (//(max_delta_x > last_delta_x && max_f_val > last_f_val)
		    //max_delta_x > last_delta_x
		    max_f_val > last_f_val
		 || max_delta_x > path_parameter.err_max_first_delta_x
		 || max_delta_x != max_delta_x) {
			success = 0;
			break;
		}

		last_delta_x = max_delta_x;
		last_f_val = max_f_val;

		update_x<<<inst.dim_grid, inst.dim_BS>>>(workspace.x, workspace.sol,
				inst.dim);
	}

	if (success) {
		if (last_delta_x > path_parameter.err_max_delta_x) {
			std::cout << "Fail tolerance: " << last_delta_x << std::endl;
			success = 0;
		}
	}

	return success;
}

bool GPU_Newton(CPUInstHom& hom, Parameter path_parameter, CT* cpu_sol0, CT cpu_t, CT*& x_new, int n_sys) {
	cout << "Newton ";
	cout << "max_it = " << path_parameter.max_it << endl;
	cout << "eps    = " << path_parameter.err_max_delta_x << endl;

	//clock_t begin = clock();

	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef, inst.n_constant, inst.n_eq, inst.dim, path_parameter.n_predictor, inst.alpha);

	workspace.update_x_t_value(cpu_sol0, cpu_t);

	clock_t begin = clock();

	bool success = newton(workspace, inst, path_parameter);

	clock_t end = clock();
	double timeSec_Newton = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );

	cout << "Path GPU Newton    Time: "<< timeSec_Newton << endl;

	x_new = workspace.get_x();

	/*clock_t end = clock();
	 double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
	 cout << "done: "<< timeSec << endl;*/
	 return success;
}

bool GPU_Path(CPUInstHom& hom, Parameter path_parameter, CT* cpu_sol0, CT cpu_t, CT*& x_gpu, int inverse, int n_sys) {
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef, inst.n_constant, \
			               inst.n_eq, inst.dim, path_parameter.n_predictor, inst.alpha);
	workspace.update_x_t(cpu_sol0, cpu_t);


	struct timeval start, end;
	long seconds, useconds;
	gettimeofday(&start, NULL);

	int n_point = 1;
	int n_step = 0;

	// Parameters
	CT delta_t = CT(path_parameter.max_delta_t,0);

	CT* tmp_t = (CT *)malloc(sizeof(CT));
	CT* tmp_t_last = (CT *)malloc(sizeof(CT));
	*tmp_t_last = cpu_t;

	int n_success = 0;

	while(tmp_t_last->real < T1(1)) {
		std::cout << "n_point = " << n_point << ", n_step = " << n_step << std::endl;

		if(delta_t.real + tmp_t_last->real < 1) {
			*tmp_t = *tmp_t_last + delta_t;
		}
		else {
			*tmp_t = CT(1,0);
		}
		std::cout << "delta_t = " << delta_t;
		std::cout << "tmp_t   = " << *tmp_t;

		if(inverse == 0){
			workspace.update_t_value(*tmp_t);
		}
		else{
			workspace.update_t_value_inverse(*tmp_t);
		}

		int n_predictor = min(workspace.n_predictor, n_point);

		std::cout << "n_predictor   = " << n_predictor << std::endl;

		int BS_pred = 32;
		int nBS_pred = (hom.dim-1)/BS_pred+1;
		std::cout << "workspace.x_t_idx = " << workspace.x_t_idx << std::endl;

		predict_newton_kernel<<<nBS_pred, BS_pred>>>(workspace.x_array, workspace.t_array,
				workspace.x, workspace.t, n_predictor, inst.dim,
				workspace.x_t_idx, n_sys);

		/*std::cout << "Predict X:" << std::endl;
		 workspace.print_x();

		 std::cout << "X Array:" << std::endl;
		 workspace.print_x_array();*/

		bool newton_success = newton(workspace, inst, path_parameter);

		if(newton_success == 1) {
			std::cout << "---------- success -----------"<< std::endl;
			n_point++;
			workspace.update_x_t_idx();
			*tmp_t_last = *tmp_t;
			n_success++;
		}
		else {
			delta_t.real = delta_t.real/2;
			std::cout << "Decrease delta_t = " << delta_t << std::endl;
			//std::cout << "      tmp_t_last = " << *tmp_t_last << std::endl;
			if(delta_t.real < path_parameter.min_delta_t) {
				break;
			}
			n_success = 0;
		}

		if(n_success > 2) {
			delta_t.real = delta_t.real*2;
			if(delta_t.real > path_parameter.max_delta_t) {
				delta_t.real = path_parameter.max_delta_t;
			}
			std::cout << "Increase delta_t = " << delta_t << std::endl;
		}

		n_step++;
		if(n_step >= path_parameter.max_step) {
			break;
		}
		std::cout << std::endl;
	}

	hom.n_step_GPU = n_step;

	bool success = 0;
	std::cout << "-------------- Path Tracking Report ---------------" << std::endl;
	if(tmp_t_last->real == 1) {
		success = 1;
		std::cout << "Success" << std::endl;
		std::cout << "n_point = " << n_point << std::endl;
		std::cout << "n_step = " << n_step << std::endl;
	}
	else {
		std::cout << "Fail" << std::endl;
		std::cout << "n_point = " << n_point << std::endl;
		std::cout << "n_step = " << n_step << std::endl;
	}

	x_gpu = workspace.get_x_last();

	gettimeofday(&end, NULL);

	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	double timeMS_Path_GPU = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	double timeSec_Path_GPU = timeMS_Path_GPU/1000;
	cout << "Path GPU Test MS   Time: "<< timeMS_Path_GPU << endl;
	cout << "Path GPU Test      Time: "<< timeSec_Path_GPU << endl;

	hom.timeSec_Path_GPU = timeSec_Path_GPU;

	return success;
}

int GPU_MGS_Large(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q,  CT*& matrix_gpu_r, int n_predictor, CT* V, int n_sys) {
	cout << "GPU Eval" << endl;

	// CUDA configuration
	cuda_set();

	/*std::cout << "bbb" << std::endl;
	GPUInst inst(hom, n_sys);
	std::cout << "ccc" << std::endl;*/

	GPUWorkspace workspace(0, 0, 0, hom.n_eq, hom.dim, 1);

	workspace.init_V_value(V);


	struct timeval start, end;
	long seconds, useconds;
	gettimeofday(&start, NULL);

	//large_MGS_origin(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1); // blocked in the bottom
	large_MGS1(workspace.V, workspace.R, workspace.P, workspace.sol, hom.n_eq, hom.dim+1);
	//large_MGS(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);

	sol_gpu = workspace.get_sol();

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	double timeMS_MGS_GPU = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	double timeSec_MGS_GPU = timeMS_MGS_GPU/1000;
	std::cout << "GPU Time MS  = " << timeMS_MGS_GPU << std::endl;
	std::cout << "GPU Time Sec = " << timeSec_MGS_GPU << std::endl;

	matrix_gpu_q = workspace.get_matrix();
	matrix_gpu_r = workspace.get_matrix_r();


	/*int rows = hom.n_eq;
	int cols = hom.dim+1;

	std::cout << "------------- Matrix Q -----------" << std::endl;
	CT* tmp_idx = matrix_gpu_q;
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			std::cout << "col=" << i << ", row=" << j << "   " << *tmp_idx++;
		}
	}*/

	cudaDeviceReset();
	return 0;
}

int GPU_Eval(const CPUInstHom& hom, CT* cpu_sol0, CT cpu_t, CT*& gpu_workspace, CT*& gpu_matrix, int n_predictor, int n_sys) {
	cout << "GPU Eval" << endl;
	// CUDA configuration
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef, inst.n_constant, inst.n_eq, inst.dim, n_predictor, inst.alpha);

	workspace.update_x_t_value(cpu_sol0, cpu_t);

	eval(workspace, inst, n_sys);
	//std::cout << "n_workspace = " << inst.n_workspace << std::endl;
	gpu_workspace = workspace.get_workspace();
	gpu_matrix = workspace.get_matrix();

	cudaDeviceReset();
	return 0;
}

int GPU_Predict(const CPUInstHom& hom, CT*& x_gpu, int n_predictor, CT t, int n_sys) {
	cout << "GPU Eval" << endl;
	std::cout << "n_predictor = " << n_predictor << std::endl;

	// CUDA configuration
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef,
	inst.n_constant, inst.n_eq, inst.dim, n_predictor, inst.alpha);

	workspace.update_t_value(t);
	workspace.init_x_t_predict_test();

	int BS = 32;
	int nBS = (hom.dim-1)/BS+1;

	std::cout << "workspace.x_t_idx = " << workspace.x_t_idx << std::endl;

	predict_newton_kernel<<<nBS, BS>>>(workspace.x_array, workspace.t_array,
	workspace.x, workspace.t, n_predictor, inst.dim, workspace.x_t_idx, n_sys);

	x_gpu = workspace.get_x();

	cudaDeviceReset();
	return 0;
}

int GPU_MGS(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q, CT*& matrix_gpu_r, int n_predictor, CT* V, int n_sys) {
	cout << "GPU Eval" << endl;

	// CUDA configuration
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef, inst.n_constant, inst.n_eq, inst.dim, n_predictor, inst.alpha);

	workspace.init_V_value(V);

	small_MGS(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);

	sol_gpu = workspace.get_sol();
	matrix_gpu_q = workspace.get_matrix();
	matrix_gpu_r = workspace.get_matrix_r();

	cudaDeviceReset();
	return 0;
}
