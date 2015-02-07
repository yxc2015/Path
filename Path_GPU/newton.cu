#include "eval.cu"
#include "mgs.cu"

__global__ void update_x_kernel(GT* x, GT* sol0, int dim)
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


__global__ void array_max_double_kernel(GT* sol, int dim, int dimLog2, double* max_delta_x ) {
	__shared__ double delta_x[max_array_size];

	int j = threadIdx.x;
	// max for the norm
	delta_x[j] = sol[j].norm_double();

	dimLog2 -= 1;
	int half_size = 1 << (dimLog2);// sum for the norm

	if(half_size > 16) {
		__syncthreads();
	}
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

bool newton(GPUWorkspace& workspace, GPUInst& inst, Parameter path_parameter) {
	bool success = 1;
	int rowsLog2 = log2ceil(inst.n_eq); // ceil for sum reduction

	double* max_delta_x_gpu;
	cudaMalloc((void **) &max_delta_x_gpu, sizeof(double));

	double* max_f_val_gpu;
	cudaMalloc((void **) &max_f_val_gpu, sizeof(double));

	double last_delta_x = path_parameter.err_max_first_delta_x;
	double last_f_val   = path_parameter.err_max_res;

	for (int i = 0; i < path_parameter.max_it; i++) {
		cout << "  Iteration " << i << endl;

		double max_delta_x;
		double max_f_val;

		eval(workspace, inst);
		inst.n_eval_GPU++;

		array_max_double_kernel<<<1, inst.n_eq>>>(workspace.f_val, inst.n_eq,
				rowsLog2, max_f_val_gpu);

		cudaMemcpy(&max_f_val, max_f_val_gpu, sizeof(double),
				cudaMemcpyDeviceToHost);

		std::cout << "       max_f_value  = " << max_f_val << std::endl;

		if (max_f_val > last_f_val || max_f_val != max_f_val) {
			success = 0;
			break;
		}

        if(max_f_val < path_parameter.err_min_round_off){
        	// last_delta_x might be problem for constant 0
        	last_delta_x = 0;
        	break;
        }

		if (inst.dim <= BS_QR) {
			mgs_small_with_delta(workspace.matrix, workspace.R, workspace.sol,
					inst.n_eq, inst.dim + 1, max_delta_x_gpu);
		} else {
			mgs_large_block(workspace.matrix, workspace.R, workspace.P, workspace.sol, inst.n_eq,\
					inst.dim + 1);
			//mgs_large(workspace.V, workspace.R, workspace.sol, inst.n_eq, inst.dim+1);

			int dimLog2 = log2ceil(inst.dim); // ceil for sum reduction
			array_max_double_kernel<<<1,inst.dim>>>(workspace.sol, inst.dim, dimLog2, max_delta_x_gpu);
		}
		inst.n_mgs_GPU++;

		cudaMemcpy(&max_delta_x, max_delta_x_gpu, sizeof(double),
				cudaMemcpyDeviceToHost);

		std::cout << "       max_delta_x  = " << max_delta_x << std::endl;

		if ( max_delta_x > last_delta_x || max_delta_x != max_delta_x) {
			success = 0;
			break;
		}

		if (max_delta_x < path_parameter.err_min_round_off) {
			last_delta_x = max_delta_x;
			break;
		}

		last_delta_x = max_delta_x;
		last_f_val = max_f_val;

		update_x_kernel<<<inst.dim_grid, inst.dim_BS>>>(workspace.x, workspace.sol,
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

