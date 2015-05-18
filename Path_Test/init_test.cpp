/*
 * init_test.cpp
 *
 *  Created on: Feb 8, 2015
 *      Author: yxc
 */

#include "init_test.h"

void Sys_filename(string& Start_Sys_filename, string& Target_Sys_file_name,\
		          int sys_type, int dim){
	std::ostringstream cyclic_filename;
	std::ostringstream cyclic_filename_target;

	if(sys_type == 0){
		cyclic_filename << "../../Problems/cyclic/cyc" << dim << "p1";
		cyclic_filename_target << "../../Problems/cyclic/cyc" << dim << "q1";
	}
	else if(sys_type == 1){
		cyclic_filename << "../../Problems/MultiPath/cyclic" << dim << ".start";
		cyclic_filename_target << "../../Problems/MultiPath/cyclic"<< dim << ".target";
	}
	else if(sys_type == 2){
		cyclic_filename << "../../Problems/PieriBig1/pieri353start" << dim-32;
		cyclic_filename_target << "../../Problems/PieriBig1/pieri353target" << dim-32;
	}
	else if(sys_type == 3){
		cyclic_filename  << "../../Problems/MultiPath/game8two.start";
		cyclic_filename_target  << "../../Problems/MultiPath/game8two.target";
	}
	else if(sys_type == 4){
		cyclic_filename  << "../../Problems/MultiPath/pieri44.start";
		cyclic_filename_target  << "../../Problems/MultiPath/pieri44.target";
	}
	else if(sys_type == 5){
		cyclic_filename  << "../../Problems/MultiPath/eq.start";
		cyclic_filename_target  << "../../Problems/MultiPath/eq.target";
	}
	else{
		cyclic_filename << "../../Problems/PieriBig2/pieri364start" << dim-32;
		cyclic_filename_target << "../../Problems/PieriBig2/pieri364target" << dim-32;
	}
	Start_Sys_filename = cyclic_filename.str();
	Target_Sys_file_name = cyclic_filename_target.str();
}


bool read_homotopy_file(PolySys& Target_Sys, PolySys& Start_Sys,\
		    int dim, int& n_eq, \
		    string Start_Sys_filename, string Target_Sys_file_name,\
		    PolySolSet* sol_set) {

	ifstream myfile(Start_Sys_filename.c_str());
	ifstream myfile_target(Target_Sys_file_name.c_str());

    if(myfile.is_open() == false){
    	std::cout << "Start System File is unable to open." << std::endl;
    	return false;
    }

    if(myfile_target.is_open() == false){
    	std::cout << "Target System File is unable to open." << std::endl;
    	return false;
    }

	VarDict pos_dict;

	Start_Sys.read_file(myfile, pos_dict);

	string x_name = "x";
	string* x_names = x_var(x_name, dim);
	Start_Sys.pos_var = x_names;

	if (dim < 8) {
		std::cout << "Start Sys" << std::endl;
		Start_Sys.print();
	}

	if(sol_set != NULL){
		sol_set -> init(myfile, pos_dict);
	}

	Target_Sys.read_file(myfile_target, pos_dict);

	string x_name_target = "x";
	string* x_names_target = x_var(x_name_target, dim);
	Target_Sys.pos_var = x_names_target;

	if (dim < 8) {
		std::cout << "Target Sys" << std::endl;
		Target_Sys.print();
	}

	n_eq = Start_Sys.n_eq;

	return true;
}

CT read_gamma(int dim){
	std::ostringstream cyclic_filename_gamma;
	cyclic_filename_gamma << "../../Problems/MultiPath/cyclic10.gamma";

	string filename_gamma = cyclic_filename_gamma.str();
	ifstream myfile_gamma(filename_gamma.c_str());

    if(myfile_gamma.is_open() == false){
    	std::cout << "Can't open gamma file." << std::endl;
    	return CT(0.0,0.0);
    }
    double gamma_real;
    myfile_gamma >> gamma_real;
    double gamma_imag;
    myfile_gamma >> gamma_imag;

    return CT(gamma_real, gamma_imag);
}

void init_cpu_inst_workspace(PolySys& Target_Sys, PolySys& Start_Sys, \
		                     int dim, int n_eq, int n_predictor, \
	                         CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, \
	                         int test){
	CT alpha;
	bool read_gamma_file = true;
	if (read_gamma_file) {
		alpha = read_gamma(10);
	}
	else {
		//srand(time(NULL));
		srand(1);
		int r = rand();
		T1 tmp = T1(r);
		alpha = CT(sin(tmp),cos(tmp));
		//alpha = CT(1,0);
		//alpha = CT(-2.65532737234004E-02,-9.99647399663787E-01);
	}
	// Fix gamma for evaluation test and
	if(test == 1){
		alpha = CT(1,0);
		//alpha = CT(1,0);
	}

	cpu_inst_hom.init(Target_Sys, Start_Sys, dim, n_eq, n_predictor, alpha);
	//cpu_inst_hom.print();
	cpu_inst_hom.init_workspace(workspace_cpu);

}

bool init_test(PolySys& Target_Sys, PolySys& Start_Sys, PolySolSet& sol_set, int dim, int& n_eq, \
		       CT*& sol0, CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, int test, int n_predictor, int sys_type){
	string Start_Sys_filename;
	string Target_Sys_file_name;
	Sys_filename(Start_Sys_filename, Target_Sys_file_name, sys_type, dim);

	std::cout << Start_Sys_filename << std::endl;
	std::cout << Target_Sys_file_name << std::endl;
	std::cout << "dim = " << dim << std::endl;

	bool read_success = read_homotopy_file(Target_Sys, Start_Sys, dim, n_eq,\
					   Start_Sys_filename, Target_Sys_file_name, &sol_set);

	if(read_success == false){
		return false;
	}

	//sol0 = rand_val_complex_n(dim);

	if(sol_set.n_sol==0){
		std::cout << "Error: No solution in start system file." << std::endl;
		return false;
	}

	sol0 = sol_set.get_sol(0);

	/*std::cout << "---------- Start Solution Top 10 ----------" << std::endl;
	for (int i = 0; i < min(dim, 10); i++) {
		std::cout << i << " " << sol0[i];
	}*/

	init_cpu_inst_workspace(Target_Sys, Start_Sys, dim, n_eq, n_predictor, \
		                    cpu_inst_hom, workspace_cpu, test);
	return true;
}

