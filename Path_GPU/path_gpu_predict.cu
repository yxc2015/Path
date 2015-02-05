// Mon block evalutation and differentiation on GPU
//template <unsigned int n_th>
__global__ void predict_newton_kernel(GT* x_array, GT* t_array, GT* x_new, GT* t_new, int n_predictor, int dim, int x_t_idx, int n_sys) {
	__shared__ GT div_diff_sh[shmemsize];
	__shared__ GT t_predictor[8]; // XXX n_predict

	int BS = blockDim.x;
	int bidx = blockIdx.x*BS;
	int tidx = threadIdx.x;
	int idx = bidx + tidx;

	/*int sys_idx = blockIdx.z;

	 x_predictor += sys_idx*n_predictor*dim;
	 t_predictor += sys_idx*n_predictor;
	 x_new += sys_idx*dim;*/

	// load t value
	if(tidx<n_predictor) {
		// XXXXXX Remove %
		t_predictor[tidx] = t_array[(tidx+x_t_idx+1)%(n_predictor+1)];
	}

	if(idx < dim) {
		GT* div_diff = div_diff_sh;
		//GT* div_diff = div_diff_sh + tidx; // XXX it can remove idx, not sure which one is better

		// Copy initial X value to divide difference
		int div_idx = 0;

		for(int np_idx = x_t_idx+1; np_idx < n_predictor+1; np_idx++) {
			div_diff[div_idx*BS + tidx]=x_array[np_idx*dim + idx];
			div_idx++;
		}

		for(int np_idx = 0; np_idx < x_t_idx; np_idx++) {
			div_diff[div_idx*BS + tidx]=x_array[np_idx*dim + idx];
			div_idx++;
		}

		// Compute divide difference
		for(int i = 1; i < n_predictor; i++) {
			for(int j = n_predictor-1; j >= i; j--) {
				div_diff[j*BS + tidx] = (div_diff[j*BS + tidx] - div_diff[(j-1)*BS + tidx])/(t_predictor[j]-t_predictor[j-i]);
			}
		}

		// Compute predict point
		GT x_tmp(0.0,0.0);
		for(int i=n_predictor-1; i > 0; i--) {
			x_tmp = (x_tmp + div_diff[i*BS + tidx]) * (*t_new - t_predictor[i-1]);
		}

		// Put X back
		x_new[idx] = x_tmp + div_diff[tidx];
	}
}

int GPU_Predict(const CPUInstHom& hom, CT*& x_gpu, int n_predictor, CT t, int n_sys) {
	cout << "GPU Eval" << endl;
	std::cout << "n_predictor = " << n_predictor << std::endl;

	// CUDA configuration
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef,
	inst.n_constant, inst.n_eq, inst.dim, n_predictor, inst.alpha);

	workspace.update_t_value(t);
	workspace.init_x_t_predict_test();

	int BS = 32;
	int nBS = (hom.dim-1)/BS+1;

	std::cout << "workspace.x_t_idx = " << workspace.x_t_idx << std::endl;

	predict_newton_kernel<<<nBS, BS>>>(workspace.x_array, workspace.t_array,
	workspace.x, workspace.t, n_predictor, inst.dim, workspace.x_t_idx, n_sys);

	x_gpu = workspace.get_x();

	cudaDeviceReset();
	return 0;
}
