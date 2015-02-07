#include "eval_mon_single.cu"

__global__ void eval_mon_seq_align_kernel(GT* workspace_mon, GT* x, GT* workspace_coef,
int* mon_pos_start_block, unsigned short* mon_pos_block, int n_mon);

void eval_mon_seq_align(GPUWorkspace& workspace, const GPUInst& inst, int n_sys){
	eval_mon_single_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
			workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
			inst.mon_pos, inst.n_mon_level[0]);

	int NB_mon = (inst.n_mon_block-1)/BS_Mon_Align + 1;
	eval_mon_seq_align_kernel<<<NB_mon, BS_Mon_Align>>>(
			workspace.mon+inst.n_mon_level[0]*2, workspace.x, workspace.coef + inst.n_mon_level[0],
			inst.mon_pos_start_block, inst.mon_pos_block,
			inst.n_mon_block);

	/*eval_mon_seq_align_block_kernel<<<inst.NB_mon_block, inst.BS_mon_block>>>(
			workspace.mon+inst.n_mon_level[0]*2, workspace.x, workspace.coef + inst.n_mon_level[0],
			inst.mon_pos_start_block, inst.mon_pos_block,
			inst.n_mon_block);*/
}

__global__ void eval_mon_seq_align_kernel(GT* workspace_mon, GT* x, GT* workspace_coef,
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


/*__global__ void eval_mon_seq_align_block_kernel(GT* workspace_mon, GT* x, GT* workspace_coef,
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
}*/
