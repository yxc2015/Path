__device__ inline int r_pos(int x, int y, int cols){
	return cols*(cols+1)/2 -1 - (y*(y+1)/2 -(x-y));
}

#include "mgs_small.cu"
#include "mgs_large.cu"

int GPU_MGS(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q,  CT*& matrix_gpu_r, int n_predictor, CT* V, int n_path) {
	cout << "GPU MGS" << endl;

	// CUDA configuration
	cuda_set();

	GPUWorkspace workspace(0, 0, 0, hom.n_eq, hom.dim, n_predictor);

	workspace.init_V_value(V);

	std::cout << "workspace.n_path = " << workspace.n_path << std::endl;

	struct timeval start, end;
	long seconds, useconds;
	gettimeofday(&start, NULL);

	if(hom.dim <= BS_QR){
		mgs_small(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);
	}
	else{
		mgs_large_block(workspace.V, workspace.R, workspace.P, workspace.sol, hom.n_eq, hom.dim+1);
	}
	//mgs_large_orig(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);
	//mgs_large_old(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);

	sol_gpu = workspace.get_sol();

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	double timeMS_MGS_GPU = seconds*1000 + useconds/1000.0;
	double timeSec_MGS_GPU = timeMS_MGS_GPU/1000;
	std::cout << "GPU Time MS  = " << timeMS_MGS_GPU << std::endl;
	std::cout << "GPU Time Sec = " << timeSec_MGS_GPU << std::endl;

	matrix_gpu_q = workspace.get_matrix();
	matrix_gpu_r = workspace.get_matrix_r();

	cudaDeviceReset();
	return 0;
}

int GPU_MGS_Mult(const CPUInstHom& hom, CT**& sol_gpu, CT**& matrix_gpu_q,  CT**& matrix_gpu_r, int n_predictor, CT* V, int n_path) {
	cout << "GPU MGS" << endl;

	// CUDA configuration
	cuda_set();

	GPUWorkspace workspace(0, 0, 0, hom.n_eq, hom.dim, 0, 0, n_path);

	std::cout << "n_path = " << n_path << std::endl;

	workspace.init_V_value(V);

	std::cout << "workspace.n_path = " << workspace.n_path << std::endl;

	sol_gpu = new CT*[n_path];
	matrix_gpu_q = new CT*[n_path];
	matrix_gpu_r = new CT*[n_path];

	struct timeval start, end;
	long seconds, useconds;
	gettimeofday(&start, NULL);

	if(hom.dim <= BS_QR){
		mgs_small1(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1, workspace.workspace_size, workspace.n_path);
	}
	else{
		mgs_large_block(workspace.V, workspace.R, workspace.P, workspace.sol, hom.n_eq, hom.dim+1);
	}

	sol_gpu[0] = workspace.get_sol(0);

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	double timeMS_MGS_GPU = seconds*1000 + useconds/1000.0;
	double timeSec_MGS_GPU = timeMS_MGS_GPU/1000;
	std::cout << "GPU Time MS  = " << timeMS_MGS_GPU << std::endl;
	std::cout << "GPU Time Sec = " << timeSec_MGS_GPU << std::endl;

	for(int path_idx=0; path_idx<n_path; path_idx++){
		sol_gpu[path_idx] = workspace.get_sol(path_idx);
		matrix_gpu_q[path_idx] = workspace.get_matrix(path_idx);
		matrix_gpu_r[path_idx] = workspace.get_matrix_r(path_idx);
	}

	cudaDeviceReset();
	return 0;
}
