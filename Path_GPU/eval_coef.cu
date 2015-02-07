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
