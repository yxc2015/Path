/*
 * path_host.cpp
 *
 *  Created on: Dec 6, 2014
 *      Author: yxc
 */

#include "path_host.h"

bool path_tracker(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, \
		          double& timeSec_Predict, double& timeSec_Eval, double& timeSec_MGS,\
		          int reverse){
	int n_point = 1;
	int n_step = 0;

	// Parameters

	CT delta_t = CT(path_parameter.max_delta_t,0);

	std::cout << "delta_t = " << delta_t;

	CT* tmp_t = workspace_cpu.t;
	CT* tmp_t_last = workspace_cpu.t_last;

	int n_success = 0;

	while(tmp_t_last->real < T1(1)){
		std::cout << "n_point = " << n_point << ", n_step = " << n_step << std::endl;

		if(delta_t.real + tmp_t_last->real < 1){
			*tmp_t = *tmp_t_last + delta_t;
		}
		else{
			*tmp_t = CT(1,0);
		}
		std::cout << "delta_t = " << delta_t;
		std::cout << "tmp_t   = " << *tmp_t;

	    clock_t begin_Predict = clock();
		int n_predictor = min(workspace_cpu.n_predictor, n_point);

		predictor_newton(workspace_cpu.x_array, workspace_cpu.t_array,\
				         workspace_cpu.x_t_idx, n_predictor, cpu_inst_hom.dim);
	    clock_t end_Predict = clock();
	    timeSec_Predict += (end_Predict - begin_Predict) / static_cast<double>( CLOCKS_PER_SEC );

		/*std::cout << "Predict X:" << std::endl;
		workspace_cpu.print_x();

	    std::cout << "X Array:" << std::endl;
		workspace_cpu.print_x_array();*/

		bool newton_success = CPU_Newton(workspace_cpu, cpu_inst_hom, path_parameter,\
				             timeSec_Eval, timeSec_MGS, reverse);

		if(newton_success == 1){
			std::cout << "---------- success -----------"<< std::endl;
			n_point++;
			workspace_cpu.update_x_t_idx();
			tmp_t = workspace_cpu.t;
			tmp_t_last = workspace_cpu.t_last;
			n_success++;
		}
		else{
			delta_t.real = delta_t.real/2;
			std::cout << "Decrease delta_t = " << delta_t << std::endl;
			if(delta_t.real < path_parameter.min_delta_t){
				break;
			}
			n_success = 0;
		}

		if(n_success > 2){
			delta_t.real = delta_t.real*2;
			if(delta_t.real > path_parameter.max_delta_t){
				delta_t.real = path_parameter.max_delta_t;
			}
			std::cout << "Increase delta_t = " << delta_t << std::endl;
		}

		n_step++;
		if(n_step >= path_parameter.max_step){
			break;
		}
		std::cout << std::endl;
	}

	cpu_inst_hom.n_step_CPU = n_step;

	bool success = 0;
	std::cout << "-------------- Path Tracking Report ---------------" << std::endl;
	if(tmp_t_last->real == 1){
		success = 1;
		std::cout << "Success" << std::endl;
		std::cout << "n_point = " << n_point << std::endl;
		std::cout << "n_step = " << n_step << std::endl;
	}
	else{
		std::cout << "Fail" << std::endl;
		std::cout << "n_point = " << n_point << std::endl;
		std::cout << "n_step = " << n_step << std::endl;
	}
	return success;
}

