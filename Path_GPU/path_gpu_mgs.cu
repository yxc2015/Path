__device__ inline int r_pos(int x, int y, int cols){
	return cols*(cols+1)/2 -1 - (y*(y+1)/2 -(x-y));
}

#include "path_gpu_mgs_small.cu"
#include "path_gpu_mgs_large.cu"

int GPU_MGS_Large(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q,  CT*& matrix_gpu_r, int n_predictor, CT* V, int n_sys) {
	cout << "GPU Eval" << endl;

	// CUDA configuration
	cuda_set();

	GPUWorkspace workspace(0, 0, 0, hom.n_eq, hom.dim, 1);

	workspace.init_V_value(V);


	struct timeval start, end;
	long seconds, useconds;
	gettimeofday(&start, NULL);

	mgs_large_block(workspace.V, workspace.R, workspace.P, workspace.sol, hom.n_eq, hom.dim+1);
	//mgs_large_orig(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);
	//mgs_large_old(workspace.V, workspace.R, workspace.sol, hom.n_eq, hom.dim+1);

	sol_gpu = workspace.get_sol();

	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	double timeMS_MGS_GPU = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	double timeSec_MGS_GPU = timeMS_MGS_GPU/1000;
	std::cout << "GPU Time MS  = " << timeMS_MGS_GPU << std::endl;
	std::cout << "GPU Time Sec = " << timeSec_MGS_GPU << std::endl;

	matrix_gpu_q = workspace.get_matrix();
	matrix_gpu_r = workspace.get_matrix_r();

	cudaDeviceReset();
	return 0;
}
