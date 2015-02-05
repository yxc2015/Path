// Mon evalutaion and differentiation on GPU
__global__ void eval_mon_global_kernel(GT* workspace_mon, GT* x, GT* workspace_coef,
int* mon_pos_start, unsigned short* mon_pos, int n_mon) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int sys_idx = blockIdx.z;
	//GT* x_d_tmp = x_d + sys_idx*dim;
	//GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

	//int tidx = threadIdx.x;
	if(idx < n_mon) {
		int tmp_start = mon_pos_start[idx];
		GT* deri = workspace_mon + tmp_start;
		unsigned short* pos = mon_pos + tmp_start;

		int n_var = pos[0];

		GT tmp = x[pos[1]];

		GT* deri_tmp = deri + 1;
		deri_tmp[1] = tmp;

		for(int i=2; i<n_var; i++) {
			tmp *= x[pos[i]];
			deri_tmp[i] = tmp;
		}

		tmp = workspace_coef[idx];

		for(int i=n_var; i>1; i--) {
			deri[i] *= tmp;
			tmp *= x[pos[i]];
		}
		deri[1] = tmp;
		deri[0] = x[pos[1]]*tmp;
	}
}
