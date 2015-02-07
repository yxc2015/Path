#ifndef __PATH_GPU_EVAL_MON_SINGLE_CU_
#define __PATH_GPU_EVAL_MON_SINGLE_CU_

// Mon evalutaion and differentiation on GPU
__global__ void eval_mon_single_kernel(GT* workspace_mon, GT* x, GT*workspace_coef,
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

		GT tmp = workspace_coef[idx];
		deri[1] = tmp;
		deri[0] = x[pos[1]]*tmp;
	}
}

#endif /*__PATH_GPU_EVAL_MON_SINGLE_CU_*/
