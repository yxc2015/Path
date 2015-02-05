#if(path_precision == 0)
#include "path_gpu_eval_mon_d.cu"
#elif(path_precision == 1)
#include "path_gpu_eval_mon_dd.cu"
#else
#include "path_gpu_eval_mon_qd.cu"
#endif


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
