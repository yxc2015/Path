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
	cpu_inst_hom.n_eval_CPU = 0;
	cpu_inst_hom.n_mgs_CPU = 0;

	CT delta_t = CT(path_parameter.max_delta_t,0);

	//std::cout << "delta_t = " << delta_t;

	CT* tmp_t = workspace_cpu.t;
	CT* tmp_t_last = workspace_cpu.t_last;

	int n_success = 0;
	string fail_reason;

    bool Debug = false;
    //Debug = true;

    if(workspace_cpu.path_idx == 0){
    	//Debug = true;
    }

    bool Record = false;
    //Record = true;

	if(Record){
    	cpu_inst_hom.path_data.add_start_pt(workspace_cpu.x_last);
	}

	while(tmp_t_last->real < T1(1)){
    	if(Debug){
    		std::cout << "n_step = " << n_step  << ", n_point = " << n_point  << std::endl;
    	}

		if(delta_t.real + tmp_t_last->real < 1){
			*tmp_t = *tmp_t_last + delta_t;
		}
		else{
			*tmp_t = CT(1,0);
		}

    	if(Debug){
			std::cout << "delta_t = " << delta_t;
			std::cout << "      t = " << *tmp_t;
    	}

	    //clock_t begin_Predict = clock();
		int n_predictor = min(workspace_cpu.n_predictor, n_point);

		predictor_newton(workspace_cpu.x_array, workspace_cpu.t_array,\
				         workspace_cpu.x_t_idx, n_predictor, cpu_inst_hom.dim);
	    //clock_t end_Predict = clock();
	    //timeSec_Predict += (end_Predict - begin_Predict) / static_cast<double>( CLOCKS_PER_SEC );

    	if(Debug){
			std::cout << "Predict" << std::endl;
			int pr = 2 * sizeof(T1);
			std::cout.precision(pr);
			for(int i=0; i<cpu_inst_hom.dim; i++){
				std::cout << i << " " << workspace_cpu.x[i];
			}
    	}

    	if(Record){
        	cpu_inst_hom.path_data.add_step_empty();
    		cpu_inst_hom.path_data.update_step_t(delta_t, *tmp_t);
        	cpu_inst_hom.path_data.update_step_predict_pt(workspace_cpu.x);
    	}

		bool newton_success = CPU_Newton(workspace_cpu, cpu_inst_hom, path_parameter,\
				             timeSec_Eval, timeSec_MGS, reverse);

		if(newton_success == 1){
	    	if(Debug){
				std::cout << "Newton Success"<< std::endl;
				int pr = 2 * sizeof(T1);
				std::cout.precision(pr);
				std::cout << "t = " << *tmp_t;
				for(int i=0; i<cpu_inst_hom.dim; i++){
					std::cout << i << " " << workspace_cpu.x[i];
				}
	    	}
			if(tmp_t->real == 1){
				CPU_Newton_Refine(workspace_cpu, cpu_inst_hom, path_parameter,\
							timeSec_Eval, timeSec_MGS, reverse);
			}
			n_point++;
			workspace_cpu.update_x_t_idx();
			tmp_t = workspace_cpu.t;
			tmp_t_last = workspace_cpu.t_last;
			n_success++;

			if(n_success > 1){
				delta_t.real = delta_t.real*path_parameter.step_increase;
				//std::cout << "Increase delta_t = " << delta_t << std::endl;
			}

			T1 max_delta_t_real;
			//std::cout << "tmp_t->real = " << tmp_t_last->real << std::endl;
			if(tmp_t_last->real > 0.9){
				max_delta_t_real = 1E-2;
			}
			else{
				max_delta_t_real = path_parameter.max_delta_t;
			}
			if(delta_t.real > max_delta_t_real){
				delta_t.real = max_delta_t_real;
			}
		}
		else{
			delta_t.real = delta_t.real*path_parameter.step_decrease;
			//std::cout << "Decrease delta_t = " << delta_t << std::endl;
			if(delta_t.real < path_parameter.min_delta_t){
				fail_reason = "delta_t too small";
				break;
			}
			n_success = 0;
		}
		n_step++;
		if(n_step >= path_parameter.max_step){
			fail_reason = "reach max step";
			break;
		}
    	if(Debug){
    		std::cout << "---------------------"<< std::endl;
    	}
		//std::cout << std::endl;
	}

	bool success = 0;
	std::cout << "-------------- Path Tracking Report ---------------" << std::endl;
	if(tmp_t_last->real == 1){
		success = 1;
		std::cout << "Success" << std::endl;
	}
	else{
		std::cout << "Fail " << fail_reason << " t=" << tmp_t_last->real<< std::endl;
	}

	if(Record){
    	cpu_inst_hom.path_data.add_end_pt(workspace_cpu.x_last);
    	cpu_inst_hom.path_data.update_success(success);
	}

	cpu_inst_hom.success_CPU = success;
	cpu_inst_hom.t_CPU = *tmp_t_last;
	cpu_inst_hom.n_step_CPU  = n_step;
	cpu_inst_hom.n_point_CPU = n_point;

	std::cout << "n_point = " << n_point << std::endl;
	std::cout << "n_step = " << n_step << std::endl;

	return success;
}

