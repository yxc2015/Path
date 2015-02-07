#if(path_precision == 0)
#include "eval_mon_d.cu"
#elif(path_precision == 1)
#include "eval_mon_dd.cu"
#else
#include "eval_mon_qd.cu"
#endif

#include "eval_sum.cu"
#include "eval_coef.cu"

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
