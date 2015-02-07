/*
 * newton_host.cpp
 *
 *  Created on: Dec 6, 2014
 *      Author: yxc
 */

#include "newton_host.h"

double CPU_normalize(complexH<double>* sol, int dim){
	double max_delta = sol[0].real*sol[0].real+sol[0].imag*sol[0].imag;
	for(int k=1; k<dim; k++){
		double tmp_delta = sol[k].real*sol[k].real+sol[k].imag*sol[k].imag;
		if(tmp_delta>max_delta){
			max_delta = tmp_delta;
		}
	}
	return max_delta;
}

double CPU_normalize(complexH<dd_real>* sol, int dim){
	double max_delta = sol[0].real.x[0]*sol[0].real.x[0]+sol[0].imag.x[0]*sol[0].imag.x[0];
	for(int k=1; k<dim; k++){
		double tmp_delta = sol[k].real.x[0]*sol[k].real.x[0]+sol[k].imag.x[0]*sol[k].imag.x[0];
		if(tmp_delta>max_delta){
			max_delta = tmp_delta;
		}
	}
	return max_delta;
}

double CPU_normalize(complexH<qd_real>* sol, int dim){
	double max_delta = sol[0].real.x[0]*sol[0].real.x[0]+sol[0].imag.x[0]*sol[0].imag.x[0];
	for(int k=1; k<dim; k++){
		double tmp_delta = sol[k].real.x[0]*sol[k].real.x[0]+sol[k].imag.x[0]*sol[k].imag.x[0];
		if(tmp_delta>max_delta){
			max_delta = tmp_delta;
		}
	}
	return max_delta;
}

bool CPU_Newton(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter,
                double& timeSec_Eval, double& timeSec_MGS, int reverse){
    cout << "Newton max_it = " << path_parameter.max_it << " err_max_delta_x = " << path_parameter.err_max_delta_x << endl;

	// Parameters
    // eqs square to compare with normal square

    //clock_t begin = clock();

    CT* x = workspace_cpu.x;
    CT t = *(workspace_cpu.t);
    CT** V = (workspace_cpu.V);
    CT** R = (workspace_cpu.R);
    CT* sol = (workspace_cpu.sol);
    int dim = cpu_inst_hom.dim;
    int n_eq = cpu_inst_hom.n_eq; // to be removed

	double last_max_delta_x = path_parameter.err_max_first_delta_x;
	double last_max_f_val   = path_parameter.err_max_res;
    bool success = 1;

    for(int i=0; i<path_parameter.max_it; i++){
        cout << "  Iteration " << i << std::endl;
	    clock_t begin_eval = clock();
    	cpu_inst_hom.eval(workspace_cpu, x, t, reverse);
    	cpu_inst_hom.n_eval_CPU++;
	    clock_t end_eval = clock();
	    timeSec_Eval += (end_eval - begin_eval) / static_cast<double>( CLOCKS_PER_SEC );

	    double max_f_val = CPU_normalize(V[dim],n_eq);
        std::cout << "       max_f_value = " << max_f_val << std::endl;

        if(max_f_val > last_max_f_val || max_f_val != max_f_val){
        	success = 0;
        	break;
        }

        if(max_f_val < path_parameter.err_min_round_off){
    		last_max_delta_x = 0;
        	break;
        }

	    clock_t begin_mgs = clock();
        CPU_mgs2qrls(V,R,sol,n_eq,dim+1);
    	cpu_inst_hom.n_mgs_CPU++;
	    clock_t end_mgs = clock();
	    timeSec_MGS += (end_mgs - begin_mgs) / static_cast<double>( CLOCKS_PER_SEC );

	    double max_delta_x = CPU_normalize(sol,dim);

        std::cout << "       max_delta_x = " << max_delta_x << std::endl;

        if(max_delta_x > last_max_delta_x || max_delta_x != max_delta_x){
        	success = 0;
        	break;
        }

        if(max_delta_x < path_parameter.err_min_round_off){
        	last_max_delta_x = max_delta_x;
        	break;
        }

        last_max_delta_x   = max_delta_x;
        last_max_f_val = max_f_val;

        for(int k=0; k<dim; k++){
        	x[k] = x[k] - sol[k];
        }
    }

    if(success){
    	if(last_max_delta_x > path_parameter.err_max_delta_x){
    		std::cout << "----------  Fail  -----------" << std::endl;
        	std::cout << "Fail tolerance: " << last_max_delta_x << std::endl;
        	success = 0;
        }
    }
    //clock_t end = clock();
    //double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
    //cout << "done: "<< timeSec << endl;
    return success;
}


