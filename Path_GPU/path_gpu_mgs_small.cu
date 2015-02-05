__global__ void mgs_small_normalize(GT* v, GT* R, \
		int dimR, int rows, int rowsLog2, int cols, int pivot );

__global__ void mgs_small_reduce(GT* v, GT* R, \
		int dimR, int rows, int rowsLog2, int cols, int pivot );

__global__ void mgs_small_backsubstitution(GT* R, GT*sol0, int dim);

__global__ void mgs_small_backsubstitution_with_delta(GT* R, GT* sol0, \
		int dim, int dimLog2, double* max_delta_x);

void mgs_small(GT* V, GT* R, GT* sol, int rows, int cols) {
	int BS = rows; // XXX Temperary solution
	//int rows = dim;
	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	int dimR = cols*(cols+1)/2;
	//int cols = dim + 1;

	/*std::cout << "rows = " << rows
	<< " cols = " << cols
	<< " rowsLog2 = " << rowsLog2 << std::endl;*/

	for(int piv=0; piv<cols-1; piv++) {
		mgs_small_normalize<<<1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
		mgs_small_reduce<<<cols-piv-1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
	}
	mgs_small_backsubstitution<<<1,cols-1>>>(R,sol,cols-1);
}

void mgs_small_with_delta(GT* V, GT* R, GT* sol, int rows, int cols, double* max_delta_x) {
	int BS = rows; // XXX Temperary solution
	//int rows = dim;
	int rowsLog2 = log2ceil(rows);// ceil for sum reduction
	int dimR = cols*(cols+1)/2;
	int dimLog2 = log2ceil(cols-1);// ceil for sum reduction

	for(int piv=0; piv<cols-1; piv++) {
		mgs_small_normalize<<<1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
		mgs_small_reduce<<<cols-piv-1,BS>>>
		(V,R,dimR,rows,rowsLog2,cols,piv);
	}
	mgs_small_backsubstitution_with_delta<<<1,cols-1>>>(R,sol,cols-1, dimLog2, max_delta_x);
}

int GPU_MGS(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q, CT*& matrix_gpu_r, int n_predictor, CT* V, int n_sys) {
	cout << "GPU Eval" << endl;

	// CUDA configuration
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef, inst.n_constant, inst.n_eq, inst.dim, n_predictor, inst.alpha);

	workspace.init_V_value(V);

	mgs_small(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);

	sol_gpu = workspace.get_sol();
	matrix_gpu_q = workspace.get_matrix();
	matrix_gpu_r = workspace.get_matrix_r();

	cudaDeviceReset();
	return 0;
}

__global__ void mgs_small_normalize(GT* v, GT* R, \
int dimR, int rows, int rowsLog2, int cols, int pivot ) {
	//int b = blockIdx.x;
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

	piv[j] /= prd[0];
	v[L] = piv[j];
	if(j == 0)
	{
		int indR = r_pos(pivot, pivot, cols); //(dimR-1) - (pivot*(pivot+1))/2 - (b*(b+1))/2 - b*(pivot+1);
		R[indR].init_imag();
		R[indR].real = prd[0];
	}
}

__global__ void mgs_small_reduce(GT* v, GT* R, \
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

__global__ void mgs_small_backsubstitution(GT* R, GT*sol0, int dim)
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

__global__ void mgs_small_backsubstitution_with_delta(GT* R, GT* sol0, int dim, int dimLog2, double* max_delta_x)
{
	int j = threadIdx.x;
	__shared__ GT sol[shmemsize/2];
	__shared__ GT Rcl[shmemsize/2];
	__shared__ double delta_x[shmemsize/2];
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

	// max for the norm
	delta_x[j] = sol[j].norm_double();
	__syncthreads();

	dimLog2 -= 1;
	int half_size = 1 << (dimLog2);// sum for the norm
	if(j + half_size < dim) {
		if(delta_x[j] < delta_x[j+half_size]) {
			delta_x[j] = delta_x[j+half_size];
		}
	}
	for(int k=0; k < dimLog2; k++) {
		if(half_size > 16) {
			__syncthreads();
		}
		half_size /= 2;
		if(j < half_size) {
			if(delta_x[j] < delta_x[j+half_size]) {
				delta_x[j] = delta_x[j+half_size];
			}
		}
	}
	if(j == 0) {
		*max_delta_x = delta_x[0];
	}
}

