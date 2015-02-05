__global__ void eval_mon_global_kernel1(GT* workspace_mon, GT* x, GT* workspace_coef,
int* mon_pos_start_block, unsigned short* mon_pos_block, int n_mon) {
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	int BS = blockDim.x;
	int idx = bidx*BS + tidx;

	if(idx < n_mon) {
		int tmp_block_start = mon_pos_start_block[bidx];
		GT* deri = workspace_mon + tmp_block_start;
		unsigned short* pos = mon_pos_block + tmp_block_start;

		int n_var = pos[tidx];

		GT tmp = x[pos[BS+tidx]];

		GT* deri_tmp = deri + BS;
		deri_tmp[BS+tidx] = tmp;

		for(int i=2; i<n_var; i++) {
			tmp *= x[pos[i*BS+tidx]];
			deri_tmp[i*BS+tidx] = tmp;
		}

		tmp = workspace_coef[idx];

		for(int i=n_var; i>1; i--) {
			deri[i*BS+tidx] *= tmp;
			tmp *= x[pos[i*BS+tidx]];
		}
		deri[BS+tidx] = tmp;
		deri[tidx] = x[pos[BS+tidx]]*tmp;
	}
}


__global__ void eval_mon_global_kernel2(GT* workspace_mon, GT* x, GT* workspace_coef,
int* mon_pos_start_block, unsigned short* mon_pos_block, int n_mon) {
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	int BS = blockDim.x;
	int idx = bidx*BS + tidx;
	int widx = idx/32;
	int wtidx = idx - widx*32;

	if(idx < n_mon) {
		int tmp_block_start = mon_pos_start_block[widx];
		GT* deri = workspace_mon + tmp_block_start;
		unsigned short* pos = mon_pos_block + tmp_block_start;

		int n_var = pos[wtidx];

		GT tmp = x[pos[32+wtidx]];

		GT* deri_tmp = deri + 32;
		deri_tmp[32+wtidx] = tmp;

		for(int i=2; i<n_var; i++) {
			tmp *= x[pos[i*32+wtidx]];
			deri_tmp[i*32+wtidx] = tmp;
		}

		tmp = workspace_coef[idx];

		for(int i=n_var; i>1; i--) {
			deri[i*32+wtidx] *= tmp;
			tmp *= x[pos[i*32+wtidx]];
		}
		deri[32+wtidx] = tmp;
		deri[wtidx] = x[pos[32+wtidx]]*tmp;
	}
}
