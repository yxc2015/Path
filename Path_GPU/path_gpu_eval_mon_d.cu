#include "path_gpu_eval_mon_single.cu"
#include "path_gpu_eval_mon_seq.cu"
#include "path_gpu_eval_mon_seq_align.cu"
#include "path_gpu_eval_mon_tree.cu"

void eval_mon(GPUWorkspace& workspace, const GPUInst& inst, int n_sys){
	int eval_method = 2;
	if(eval_method == 0){
		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
				inst.mon_pos, inst.n_mon_level[0]);

		eval_mon_global_kernel<<<inst.mon_global_grid, inst.mon_global_BS>>>(
				workspace.mon, workspace.x, workspace.coef + inst.n_mon_level[0],
				inst.mon_pos_start + inst.n_mon_level[0], inst.mon_pos,
				inst.n_mon_global);
	}
	else if(eval_method == 1){
		std::cout << "inst.level = " << inst.level << std::endl;
		int max_level = 9;

		int* pos_start_tmp = inst.mon_pos_start;
		GT* workspace_coef_tmp = workspace.coef;

		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
				inst.mon_pos, inst.n_mon_level[0]);

		pos_start_tmp += inst.n_mon_level[0];
		workspace_coef_tmp += inst.n_mon_level[0];

		if(inst.level > 1){
			int n_mon_tmp = inst.n_mon_level[1];
			mon_block_unroll2<<<inst.mon_level_grid[1], inst.mon_global_BS>>>(
					workspace.mon, workspace.x, workspace_coef_tmp,
					pos_start_tmp, inst.mon_pos,
					n_mon_tmp);
			pos_start_tmp += n_mon_tmp;
			workspace_coef_tmp += n_mon_tmp;
		}

	    if(inst.level > 2){
			int n_mon_tmp = inst.n_mon_level[2];
			mon_block_unroll_4<<<inst.mon_level_grid[2], inst.mon_level_BS>>>(
					workspace.mon, workspace.x, workspace_coef_tmp,
					pos_start_tmp, inst.mon_pos,
					n_mon_tmp);
			pos_start_tmp += n_mon_tmp;
			workspace_coef_tmp += n_mon_tmp;
		}

	    int last_level = min(inst.level, max_level+1);
	    for(int i=3; i<last_level; i++){
			if(i==3){
				mon_block_unroll<4><<<inst.mon_level_grid[3], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==4){
				mon_block_unroll<8><<<inst.mon_level_grid[4], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==5){
				mon_block_unroll<16><<<inst.mon_level_grid[5], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==6){
				mon_block_unroll<32><<<inst.mon_level_grid[6], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==7){
				mon_block_unroll<64><<<inst.mon_level_grid[7], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==8){
				mon_block_unroll<128><<<inst.mon_level_grid[8], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==9){
				mon_block_unroll<256><<<inst.mon_level_grid[9], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			pos_start_tmp += inst.n_mon_level[i];
			workspace_coef_tmp += inst.n_mon_level[i];
	    }

	    // To be tested and improved by tree structure by sequential bottom level
	    if(inst.n_mon_level_rest[last_level-1] > 0){
			eval_mon_global_kernel<<<inst.mon_level_grid_rest[last_level-1], inst.mon_global_BS>>>(
					workspace.mon, workspace.x, workspace_coef_tmp,
					pos_start_tmp, inst.mon_pos, inst.n_mon_level_rest[last_level-1]);
	    }
    }
	else if(eval_method == 2){
		std::cout << "Eval NEW" << std::endl;
		std::cout << "inst.level = " << inst.level << std::endl;
		int max_level = 9;

		int* pos_start_tmp = inst.mon_pos_start;
		GT* workspace_coef_tmp = workspace.coef;

	    int last_level = min(inst.level, max_level+1);
	    for(int i=0; i<last_level; i++){
	    	if(i==0){
	    		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
	    				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
	    				inst.mon_pos, inst.n_mon_level[0]);
	    	}
	    	else if(i==1){
				mon_block_unroll2<<<inst.mon_level_grid[1], inst.mon_global_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[1]);

	    	}
	    	else if(i==2){
				mon_block_unroll_4<<<inst.mon_level_grid[2], inst.mon_level_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
	    	}
	    	else if(i==3){
				mon_block_unroll<4><<<inst.mon_level_grid[3], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
	    	else if(i==4){
				mon_block_unroll<8><<<inst.mon_level_grid[4], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
	    	else if(i==5){
				mon_block_unroll<16><<<inst.mon_level_grid[5], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
	    	else if(i==6){
				mon_block_unroll<32><<<inst.mon_level_grid[6], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
	    	else if(i==7){
				mon_block_unroll<64><<<inst.mon_level_grid[7], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
	    	else if(i==8){
				mon_block_unroll<128><<<inst.mon_level_grid[8], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
	    	else if(i==9){
				mon_block_unroll<256><<<inst.mon_level_grid[9], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			pos_start_tmp += inst.n_mon_level[i];
			workspace_coef_tmp += inst.n_mon_level[i];
	    }

	    if(inst.level > max_level+1){
			int n_mon_tmp = inst.n_mon_level_rest[max_level];
	    	if(max_level == 1){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_global_BS, n_sys, 1);
				eval_mon_global_kernel<<<mon_level_rest_grid, inst.mon_global_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, n_mon_tmp);

	    	}
	    	else if(max_level == 2){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 2);
				mon_block_unroll_4n<<<mon_level_rest_grid, inst.mon_level_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 3){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 4);
				mon_block_unroll_n<4><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 4){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 8);
				mon_block_unroll_n<8><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 5){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 16);
				mon_block_unroll_n<16><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	if(max_level == 6){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 32);
				mon_block_unroll_n<32><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 7){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 64);
				mon_block_unroll_n<64><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 8){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 128);
				mon_block_unroll_n<128><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 9){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 256);
				mon_block_unroll_n<256><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    }
	}
	else{
		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
				inst.mon_pos, inst.n_mon_level[0]);

		/*eval_mon_global_kernel1<<<inst.NB_mon_block, inst.BS_mon_block>>>(
				workspace.mon+inst.n_mon_level[0]*2, workspace.x, workspace.coef + inst.n_mon_level[0],
				inst.mon_pos_start_block, inst.mon_pos_block,
				inst.n_mon_block);*/
		int BS_mon = 64;
		int NB_mon = (inst.n_mon_block-1)/BS_mon + 1;
		eval_mon_global_kernel2<<<NB_mon, BS_mon>>>(
				workspace.mon+inst.n_mon_level[0]*2, workspace.x, workspace.coef + inst.n_mon_level[0],
				inst.mon_pos_start_block, inst.mon_pos_block,
				inst.n_mon_block);
	}
}
