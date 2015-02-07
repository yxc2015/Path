#ifndef __PATH_GPU_EVAL_SUM_SEQ_CU_
#define __PATH_GPU_EVAL_SUM_SEQ_CU_

// Mon evalutaion and differentiation on GPU
__global__ void eval_sum_seq_kernel(GT* workspace_matrix, GT* workspace_sum, int* sum_pos, int* sum_pos_start, int n_sum, int n_sys) {
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

void eval_sum_seq(GPUWorkspace& workspace, const GPUInst& inst, int n_sys){
		eval_sum_seq_kernel<<<inst.sum_grid, inst.sum_BS>>>(workspace.matrix,
				workspace.sum, inst.sum_pos, inst.sum_pos_start, inst.n_sum, n_sys);
}

#endif /*__PATH_GPU_EVAL_SUM_SEQ_CU_*/
