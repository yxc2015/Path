#include "cuda_set.cu"

#include "parameter.h"
#include "complex.cu"

#include "path_gpu_data.cu"

#include "predict.cu"
#include "newton.cu"

bool path(GPUWorkspace& workspace, GPUInst& inst, Parameter path_parameter, CT cpu_t, int n_sys, int inverse = 0) {
	int n_point = 1;
	int n_step = 0;

	// Parameters
	CT delta_t = CT(path_parameter.max_delta_t,0);

	CT* tmp_t = (CT *)malloc(sizeof(CT));
	CT* tmp_t_last = (CT *)malloc(sizeof(CT));
	*tmp_t_last = cpu_t;

	int n_success = 0;

	while(tmp_t_last->real < T1(1)) {
		std::cout << "n_point = " << n_point << ", n_step = " << n_step << std::endl;

		if(delta_t.real + tmp_t_last->real < 1) {
			*tmp_t = *tmp_t_last + delta_t;
		}
		else {
			*tmp_t = CT(1,0);
		}
		std::cout << "delta_t = " << delta_t;
		std::cout << "tmp_t   = " << *tmp_t;

		if(inverse == 0){
			workspace.update_t_value(*tmp_t);
		}
		else{
			workspace.update_t_value_inverse(*tmp_t);
		}

		int n_predictor = min(workspace.n_predictor, n_point);

		std::cout << "n_predictor   = " << n_predictor << std::endl;

		int BS_pred = 32;
		int nBS_pred = (inst.dim-1)/BS_pred+1;
		std::cout << "workspace.x_t_idx = " << workspace.x_t_idx << std::endl;

		predict_newton_kernel<<<nBS_pred, BS_pred>>>(workspace.x_array, workspace.t_array,
				workspace.x, workspace.t, n_predictor, inst.dim,
				workspace.x_t_idx, n_sys);

		/*std::cout << "Predict X:" << std::endl;
		 workspace.print_x();

		 std::cout << "X Array:" << std::endl;
		 workspace.print_x_array();*/

		bool newton_success = newton(workspace, inst, path_parameter);

		if(newton_success == 1) {
			std::cout << "---------- success -----------"<< std::endl;
			n_point++;
			workspace.update_x_t_idx();
			*tmp_t_last = *tmp_t;
			n_success++;
		}
		else {
			delta_t.real = delta_t.real/2;
			std::cout << "Decrease delta_t = " << delta_t << std::endl;
			//std::cout << "      tmp_t_last = " << *tmp_t_last << std::endl;
			if(delta_t.real < path_parameter.min_delta_t) {
				break;
			}
			n_success = 0;
		}

		if(n_success > 2) {
			delta_t.real = delta_t.real*2;
			if(delta_t.real > path_parameter.max_delta_t) {
				delta_t.real = path_parameter.max_delta_t;
			}
			std::cout << "Increase delta_t = " << delta_t << std::endl;
		}

		n_step++;
		if(n_step >= path_parameter.max_step) {
			break;
		}
		std::cout << std::endl;
	}

	bool success = 0;
	std::cout << "-------------- Path Tracking Report ---------------" << std::endl;
	if(tmp_t_last->real == 1) {
		success = 1;
		std::cout << "Success" << std::endl;
	}
	else {
		std::cout << "Fail" << std::endl;
	}

	inst.n_step_GPU = n_step;
	inst.n_point_GPU = n_point;
	return success;
}

bool GPU_Path(CPUInstHom& hom, Parameter path_parameter, CT* cpu_sol0, CT cpu_t, CT*& x_gpu, int inverse, int n_sys) {
	cuda_set();

	GPUInst inst(hom, n_sys);
	GPUWorkspace workspace(inst.n_workspace, inst.n_coef, inst.n_constant, \
			               inst.n_eq, inst.dim, path_parameter.n_predictor, inst.alpha);
	workspace.update_x_t(cpu_sol0, cpu_t);

	struct timeval start, end;
	long seconds, useconds;
	gettimeofday(&start, NULL);

	bool success = path(workspace, inst, path_parameter, cpu_t, n_sys, inverse);
	x_gpu = workspace.get_x_last();

	gettimeofday(&end, NULL);

	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	double timeMS_Path_GPU = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	double timeSec_Path_GPU = timeMS_Path_GPU/1000;

	cout << "Path GPU Test MS   Time: "<< timeMS_Path_GPU << endl;
	cout << "Path GPU Test      Time: "<< timeSec_Path_GPU << endl;
	cout << "Path GPU Step     Count: "<< inst.n_step_GPU << endl;
	cout << "Path GPU Point    Count: "<< inst.n_point_GPU << endl;
	cout << "Path GPU Eval     Count: "<< inst.n_eval_GPU << endl;
	cout << "Path GPU MGS      Count: "<< inst.n_mgs_GPU << endl;

	hom.timeSec_Path_GPU = timeSec_Path_GPU;
	hom.n_step_GPU = inst.n_step_GPU;
	hom.n_point_GPU = inst.n_point_GPU;
	hom.n_eval_GPU = inst.n_eval_GPU;
	hom.n_mgs_GPU = inst.n_mgs_GPU;

	return success;
}
