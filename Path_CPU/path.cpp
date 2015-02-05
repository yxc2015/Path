/** @file */
#include <iostream>
#include <sstream>

#include "families.h"
#include "path_host.h"
#include "path_gpu.h"
#include "newton_host.h"
#include "parameter.h"
#include <ctime>

void generate_cyclic_system(PolySys& Target_Sys, PolySys& Start_Sys, PolySolSet& sol_set, int dim);

T1 eval_test_classic(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, CT* sol0, CT t, PolySys& Target_Sys, int n_eq, int dim);

T1 predict_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, CT t);

T1 mgs_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom);

T1 mgs_test_large(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom);

T1 mgs_test_any(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, int device_option = 0);

T1 newton_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, CT t);

bool path_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0,  CT*& sol_new, CT t, PolySys& Target_Sys, int device_option = 0);

T1 path_test_reverse(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, CT t, int device_option = 1);

int witness_set_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, CT t, int device_option = 1);

void all_test(PolySys& Target_Sys, Workspace& workspace_cpu,
			  CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, int n_eq, int dim);

T1 err_check_workspace(const CT* workspace1, const CT* workspace2, int n_workspace_size, int n_err_print = 20);

T1 err_check_workspace_matrix(const CT* workspace1, const CT* workspace2, int n_rows, int n_cols);

void generate_monodromy(PolySys& Target_Sys, PolySys& Start_Sys,\
		int dim, int& n_eq, \
		string Start_Sys_filename, string Target_Sys_file_name, \
		PolySolSet* sol_set=NULL);

void Pieri_Test(int dim_start, int dim_end, Parameter path_parameter, int sys_type, int device_option);

void init_test(PolySys& Target_Sys, PolySys& Start_Sys, PolySolSet& sol_set, int dim, int& n_eq, \
		       CT*& sol0, CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, int test, int n_predictor, int sys_type = 2);

class Sol{
public:
	int dim;
	CT* sol;
	Sol(CT* sol, int dim){
		this->dim = dim;
		this->sol = new CT[dim];
		for(int i=0; i<dim; i++){
			this->sol[i] = sol[i];
		}
	}

	bool operator == (const Sol& that){
		T1 dis = 0;
		for(int i=0; i<dim; i++){
			T1 tmp_dis = abs(sol[i].real - that.sol[i].real);
			if(tmp_dis > dis){
				dis = tmp_dis;
			}
			tmp_dis = abs(sol[i].imag - that.sol[i].imag);
			if(tmp_dis > dis){
				dis = tmp_dis;
			}
		}
		if(dis > 1E-5){
			return 0;
		}
		return 1;
	}

	~Sol(){
		delete[] sol;
	}

	void print(){
		std::cout << "dim = " << dim << std::endl;
		for(int i=0; i<dim; i++){
			std::cout << i << " " << sol[i];
		}
		std::cout << std::endl;
	}
};

class SolSet{
public:
	int dim;
	int n;
	Sol** sols;

	SolSet(int dim){
		this->dim = dim;
		n = 0;
		sols = new Sol*[100];
		for(int i=0; i<100; i++){
			sols[i] = NULL;
		}
	}

	bool add(CT* new_sol){
		Sol* tmp_sol= new Sol(new_sol, dim);
		for(int i=0; i<n; i++){
			if(*tmp_sol == *sols[i]){
				return 0;
			}
		}
		std::cout << "Add New Solution" << std::endl;
		sols[n] = tmp_sol;
		n++;
		return 1;
	}

	void print(){
		std::cout << "n_sol = " << n << std::endl;
		for(int i=0; i<n; i++){
			sols[i]->print();
		}
	}

	void print_short(){
		std::cout << "n_sol = " << n << std::endl;
		for(int i=0; i<n; i++){
			std::cout << i << " x[0] = " << sols[i]->sol[0];
		}
	}
};

int main_test(int test, int dim, int device_option) {
	//int test = 7;
	// 0: Predict Test
	// 1: Evalute Test
	// 2: Modified Gram-Smith Test
	// 3: Modified Gram-Smith Large Test
	// 4: Modified Gram-Smith Large Test for any dim
	// 5: Newton Test
	// 6: Path Tracking Test
	// 7: Path Tracking Reverse Test
	// 8: Witeness Set Test
	// 9: Pieri Test
	// else: All Test
	Parameter path_parameter(N_PREDICTOR, MAX_STEP, MAX_IT, MAX_DELTA_T, MIN_DELTA_T,\
			                 ERR_MAX_RES, ERR_MAX_DELTA_X, \
			                 ERR_MAX_FIRST_DELTA_X, ERR_MIN_ROUND_OFF);

	int pr = 2 * sizeof(T1);
	std::cout.precision(pr);

	if(test != 9){
		int sys_type = 0;
		int n_eq;
		CT* sol0;
		PolySys Target_Sys;
		PolySys Start_Sys;
		PolySolSet sol_set;
		CPUInstHom cpu_inst_hom;
		Workspace workspace_cpu;

		if(test != 4){
			init_test(Target_Sys, Start_Sys, sol_set, dim, n_eq, sol0, cpu_inst_hom, \
					  workspace_cpu, test, path_parameter.n_predictor, sys_type);
		}
		else{
			int n_eq = dim;
			cpu_inst_hom.dim = dim;
			cpu_inst_hom.n_eq = n_eq;
			int workspace_size = 5;
			int n_coef = 1;
			int n_constant = 1;
			int n_predictor = 1;
			workspace_cpu.init(workspace_size, n_coef, n_constant, n_eq, dim, n_predictor);
            sol0 = new CT[dim];
		}

		if (test == 0) {
			// Predict Test
			CT t(0.5,0);
			predict_test(workspace_cpu, cpu_inst_hom, t);
		}
		else if(test == 1) {
			// Evaluate Test
			CT t(T1(1),T1(0));
			eval_test_classic(workspace_cpu, cpu_inst_hom, sol0, t, Target_Sys, n_eq, dim);
		}
		else if(test == 2) {
			// Modified Gram-Smith Test
			CT t = CT(0.1,0);
			workspace_cpu.update_x_t_value(sol0, t);
			mgs_test(workspace_cpu, cpu_inst_hom);
		}
		else if(test == 3) {
			// Modified Gram-Smith Test
			CT t = CT(0.1,0);
			workspace_cpu.update_x_t_value(sol0, t);
			mgs_test_large(workspace_cpu, cpu_inst_hom);
		}
		else if(test == 4) {
			// Modified Gram-Smith Large Test for any dim
			CT t = CT(0.1,0);
			workspace_cpu.update_x_t_value(sol0, t);
			mgs_test_any(workspace_cpu, cpu_inst_hom, device_option);
		}
		else if(test == 5) {
			// Newton Test
			CT t = CT(0.0001,0);
			workspace_cpu.update_x_t_value(sol0, t);
			newton_test(workspace_cpu, cpu_inst_hom, path_parameter, sol0, t);
		}
		else if(test == 6) {
			// Path Tracking Test
			CT t(0.0,0.0);
			workspace_cpu.update_x_t(sol0, t);
			CT* sol_new = NULL;
			path_test(workspace_cpu, cpu_inst_hom, path_parameter, sol0, sol_new, t, Target_Sys, device_option);
		}
		else if(test == 7) {
			// Path Tracking Reverse Test
			CT t(0.0,0.0);
			workspace_cpu.update_x_t(sol0, t);
			path_test_reverse(workspace_cpu, cpu_inst_hom, path_parameter, sol0, t, device_option);
		}
		else if(test == 8) {
			// Path Tracking Reverse Test
			CT t(0.0,0.0);
			workspace_cpu.update_x_t(sol0, t);
			witness_set_test(workspace_cpu, cpu_inst_hom, path_parameter, sol0, t, device_option);
		}
		else {
			all_test(Target_Sys, workspace_cpu, cpu_inst_hom, path_parameter, sol0, n_eq, dim);
		}
		//delete[] sol0;
	}
	else{
		int dim_start = 32;
		int dim_end = 36;
		int sys_type = 3;
		Pieri_Test(dim_start, dim_end, path_parameter, sys_type, device_option);
	}
	return 0;
}

int parse_arguments
 ( int argc, char *argv[], int& test, int& dim, int& device_option )
/* Parses the arguments on the command line.
   Returns 0 if okay, otherwise the function returns 1. */
{
	if(argc < 4){
		cout << argv[0] << " needs 3 parameters: test, dim, device_option" << endl;
		cout << "Test  option: " << std::endl
		     << "  0: Predict Test" << endl \
		     << "  1: Evalute Test" << endl  \
		     << "  2: Modified Gram-Smith Test" << endl  \
		     << "  3: Modified Gram-Smith Large Test" << endl  \
		     << "  4: Modified Gram-Smith Large Test for Any Dim" << endl  \
		     << "  5: Newton Test" << endl  \
		     << "  6: Path Tracking Test" << endl  \
		     << "  7: Path Tracking Reverse Test" << endl  \
		     << "  8: Witeness Set Test" << endl  \
		     << "  9: Pieri Test" << endl  \
		     << "  else: All Test" << endl;
		cout << "Device option: " << std::endl \
			 << "  0. CPU and GPU (only for test 5 Path Test, 8 Pieri Test)" << std::endl \
			 << "  1. CPU         (only for test 5 Path Test, 6: Path Reverse Test, 7 Witeness Set Test, 8 Pieri Test)" << std::endl \
			 << "  2. GPU         (only for test 5 Path Test, 6: Path Reverse Test, 7 Witeness Set Test, 8 Pieri Test)" << std::endl \
			 << "All other test runs both on CPU and GPU." << device_option << std::endl;
		cout << "please try again..." << endl; return 1;
   }
   test = atoi(argv[1]);     // block size
   dim = atoi(argv[2]);    // dimension
   device_option = atoi(argv[3]);     // number of monomials
   return 0;
}


int main ( int argc, char *argv[] )
{
   // Initialization of the execution parameters

   int test,dim,device_option;
   if(parse_arguments(argc,argv,test,dim,device_option) == 1) return 1;

   main_test(test,dim,device_option);
}
void write_complex_array(string file_name, CT* array, int dim){
	ofstream tfile(file_name.c_str());

	int pr = 2 * sizeof(T1);
	tfile.precision(pr+10);
	tfile<<fixed;

	for(int i=0; i<dim; i++){
		tfile << array[i].real << " "
			  << array[i].imag << std::endl;
	}
	tfile.close();
};

CT* read_complex_array(string file_name, int dim){
	CT* array = new CT[dim];
	ifstream tfile(file_name.c_str());
	for(int i=0; i<dim; i++){
		array[i] = get_complex_number(tfile);
	}
	tfile.close();
	return array;
}

void Sys_filename(string& Start_Sys_filename, string& Target_Sys_file_name,\
		          int sys_type, int dim){
	std::ostringstream cyclic_filename;
	std::ostringstream cyclic_filename_target;

	if(sys_type == 0){
		cyclic_filename << "../Problems/cyclic/cyc" << dim << "p1";
		cyclic_filename_target << "../Problems/cyclic/cyc" << dim << "q1";
	}
	else if(sys_type == 1){
		cyclic_filename << "../Problems/PieriBig1/pieri42start0";
		cyclic_filename_target << "../Problems/PieriBig1/pieri42target0";
	}
	else if(sys_type == 2){
		cyclic_filename << "../Problems/PieriBig1/pieri353start" << dim-32;
		cyclic_filename_target << "../Problems/PieriBig1/pieri353target" << dim-32;
	}
	else{
		cyclic_filename << "../Problems/PieriBig2/pieri364start" << dim-32;
		cyclic_filename_target << "../Problems/PieriBig2/pieri364target" << dim-32;
	}
	Start_Sys_filename = cyclic_filename.str();
	Target_Sys_file_name = cyclic_filename_target.str();
}

string sol_filename(int dim, int sys_type = 2){
	std::ostringstream sol_filename;

	if(sys_type == 0){
		sol_filename << "../Problems/cyclic/cyc" << dim << "q1" << ".sol";
	}
	else if(sys_type == 2){
		sol_filename << "../Problems/PieriBig1/pieri353target"<< dim-32 << ".sol";
	}
	else if(sys_type == 3){
		sol_filename << "../Problems/PieriBig2/pieri364target"<< dim-32 << ".sol";
	}
	return sol_filename.str();
}

void init_cpu_inst_workspace(PolySys& Target_Sys, PolySys& Start_Sys, \
		                     int dim, int n_eq, int n_predictor, \
	                         CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, \
	                         int test){
	CT alpha;
	if (test != 9) {
		//srand(time(NULL));
		srand(1);
		int r = rand();
		T1 tmp = T1(r);
		alpha = CT(sin(tmp),cos(tmp));
		//alpha = CT(1,0);
	}
	else {
		alpha = CT(1,0);
	}

	cpu_inst_hom.init(Target_Sys, Start_Sys, dim, n_eq, n_predictor, alpha);
	//cpu_inst_hom.print();
	cpu_inst_hom.init_workspace(workspace_cpu);

}

void init_test(PolySys& Target_Sys, PolySys& Start_Sys, PolySolSet& sol_set, int dim, int& n_eq, \
		       CT*& sol0, CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, int test, int n_predictor, int sys_type){
	string Start_Sys_filename;
	string Target_Sys_file_name;
	Sys_filename(Start_Sys_filename, Target_Sys_file_name, sys_type, dim);

	std::cout << Start_Sys_filename << std::endl;
	std::cout << Target_Sys_file_name << std::endl;
	std::cout << "dim = " << dim << std::endl;

	if((sys_type != 2 && sys_type != 3) || dim == 32){
		generate_monodromy(Target_Sys, Start_Sys, dim, n_eq,\
						   Start_Sys_filename, Target_Sys_file_name, &sol_set);

		//sol0 = rand_val_complex_n(dim);
		sol0 = sol_set.get_sol(0);
	}
	else{
		generate_monodromy(Target_Sys, Start_Sys, dim, n_eq,\
						   Start_Sys_filename, Target_Sys_file_name);
		string pieri_sol_filename = sol_filename(dim-1, sys_type);
		std::cout << "pieri_sol_filename = " << pieri_sol_filename << std::endl;
		sol0 = read_complex_array(pieri_sol_filename, dim-1);
		CT* sol_new = new CT[dim];
		for(int i=0; i<dim-1; i++){
			sol_new[i] = sol0[i];
		}
		sol_new[dim-1] = CT(0.0,0);
		delete[] sol0;
		sol0 = sol_new;
	}

	std::cout << "---------- Start Solution Top 10 ----------" << std::endl;
	for (int i = 0; i < min(dim, 10); i++) {
		std::cout << i << " " << sol0[i];
	}

	init_cpu_inst_workspace(Target_Sys, Start_Sys, dim, n_eq, n_predictor, \
		                    cpu_inst_hom, workspace_cpu, test);
}


void init_test_pieri(PolySys& Target_Sys, PolySys& Start_Sys, PolySolSet& sol_set, int dim, int& n_eq, \
		       CT*& sol0, CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, int test, int n_predictor, int sys_type = 2){
	string Start_Sys_filename;
	string Target_Sys_file_name;

	Sys_filename(Start_Sys_filename, Target_Sys_file_name, sys_type, dim);

	if(dim == 32){
		generate_monodromy(Target_Sys, Start_Sys, dim, n_eq,\
				           Start_Sys_filename, Target_Sys_file_name, &sol_set);
		sol0 = sol_set.get_sol(0);
	}
	else{
		generate_monodromy(Target_Sys, Start_Sys, dim, n_eq,\
				           Start_Sys_filename, Target_Sys_file_name);
		if(sol0 == NULL){
			string pieri_sol_filename = sol_filename(dim-1, sys_type);
			sol0 = read_complex_array(pieri_sol_filename, dim-1);

		}
		CT* sol_new = new CT[dim];
		for(int i=0; i<dim-1; i++){
			sol_new[i] = sol0[i];
		}
		sol_new[dim-1] = CT(0.0,0);
		delete[] sol0;
		sol0 = sol_new;
	}

	init_cpu_inst_workspace(Target_Sys, Start_Sys, dim, n_eq, n_predictor, \
		                    cpu_inst_hom, workspace_cpu, test);
}

void Pieri_Test(int dim_start, int dim_end, Parameter path_parameter, int sys_type, int device_option){
	CT* sol0 = NULL;
	int test = 7;

	std::ostringstream file_name;
	file_name << "../Problems/pieri_timing_" << sys_type-1;
	ofstream tfile(file_name.str().c_str());

	int pr = 2 * sizeof(T1);
	tfile.precision(pr+10);
	tfile<<fixed;

	int dim = dim_start;

	for(dim=dim_start; dim<dim_end; dim++){
		int n_eq;
		PolySys Target_Sys;
		PolySys Start_Sys;
		PolySolSet sol_set;
		CPUInstHom cpu_inst_hom;
		Workspace workspace_cpu;

		init_test_pieri(Target_Sys, Start_Sys, sol_set, dim, n_eq, sol0, cpu_inst_hom, \
				  workspace_cpu, test, path_parameter.n_predictor, sys_type);

		// Path Tracking Test
		CT t(0.0,0.0);
		workspace_cpu.init_x_t_idx();
		workspace_cpu.update_x_t(sol0, t);
		CT* sol_new = NULL;
		bool success = path_test(workspace_cpu, cpu_inst_hom, path_parameter, sol0, sol_new, t, Target_Sys, device_option);
		if(success == 0){
			std::cout << "Pieri Test Fail!" << std::endl;
			std::cout << "dim = " << dim << std::endl;
			break;
		}
		for(int i=0; i<dim; i++){
			sol0[i] = sol_new[i];
		}

		//free gpu solution memory
		if(device_option==0 or device_option==2){
			free(sol_new);
		}
		/*CT* f_val = Target_Sys.eval(sol0);
		for(int i=0; i<n_eq; i++){
			std::cout << i << " " << f_val[i] << std::endl;
		}*/

		string pieri_sol_filename = sol_filename(dim, sys_type);
		write_complex_array(pieri_sol_filename, sol0, dim);

		/*delete[] sol0;
		sol0 = read_complex_array(pieri_sol_filename.str(), dim);
		for(int i=0; i<dim; i++){
			std::cout << sol0[i];
		}*/
		tfile << dim << " " << cpu_inst_hom.success_CPU \
				     << " " << cpu_inst_hom.success_GPU \
				     << " " << cpu_inst_hom.n_step_CPU \
				     << " " << cpu_inst_hom.n_step_GPU \
				     << " " << cpu_inst_hom.timeSec_Path_CPU \
				     << " " << cpu_inst_hom.timeSec_Path_GPU << std::endl;
	}

	if(dim == dim_end){
		std::cout << "Pieri Test Success!" << std::endl;
		std::cout << "dim = " << dim << std::endl;
	}

	tfile.close();
}


void generate_monodromy(PolySys& Target_Sys, PolySys& Start_Sys,\
		    int dim, int& n_eq, \
		    string Start_Sys_filename, string Target_Sys_file_name,\
		    PolySolSet* sol_set) {
	VarDict pos_dict;

	ifstream myfile(Start_Sys_filename.c_str());

	Start_Sys.read_file(myfile, pos_dict);

	string x_name = "x";
	string* x_names = x_var(x_name, dim);
	Start_Sys.pos_var = x_names;

	if (dim < 32) {
		Start_Sys.print();
	}

	if(sol_set != NULL){
		sol_set -> init(myfile);
	}

	VarDict pos_dict_target;

	ifstream myfile_target(Target_Sys_file_name.c_str());

	Target_Sys.read_file(myfile_target, pos_dict_target);

	string x_name_target = "x";
	string* x_names_target = x_var(x_name_target, dim);
	Target_Sys.pos_var = x_names_target;

	if (dim < 32) {
		Target_Sys.print();
	}

	n_eq = Start_Sys.n_eq;
}

void all_test(PolySys& Target_Sys, Workspace& workspace_cpu, \
			  CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0,  int n_eq, int dim) {

	// Predict Test
	T1* err = new T1[5];

	CT t(0.5,0);
	err[0] = predict_test(workspace_cpu, cpu_inst_hom, t);

	// Evaluate Test
	t = CT(1,0);
	workspace_cpu.update_x_t_value(sol0, t);
	err[1] = eval_test_classic(workspace_cpu, cpu_inst_hom, sol0, t, Target_Sys, n_eq, dim);

	// Modified Gram-Smith Test
	t = CT(0.1,0);
	workspace_cpu.update_x_t_value(sol0, t);
	err[2] = mgs_test(workspace_cpu, cpu_inst_hom);

	// Newton Test
	t = CT(0.1,0);
	workspace_cpu.update_x_t_value(sol0, t);
	err[3] = newton_test(workspace_cpu, cpu_inst_hom, path_parameter, sol0, t);

	// Path Tracking Test
	t = CT(0.0,0.0);
	workspace_cpu.update_x_t(sol0, t);
	CT* sol_new = NULL;
	bool path_success = path_test(workspace_cpu, cpu_inst_hom, path_parameter, sol0, sol_new, t, Target_Sys);

	std::cout << "--------- Test Error Report ----------" << std::endl;
	std::cout << "Predict : " << err[0] << std::endl;
	std::cout << "Eval    : " << err[1] << std::endl;
	std::cout << "MGS     : " << err[2] << std::endl;
	std::cout << "Newton  : " << err[3] << std::endl;
	std::cout << "Path    : " << path_success << std::endl;

	delete[] err;
}

T1 path_test_reverse(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, CT t, int device_option) {
	std::cout << "--------- Path Tracking Reverse Test ----------" << std::endl;

	bool cpu_test = 0;
	bool gpu_test = 0;
	if(device_option == 1){
			cpu_test = 1;
			gpu_test = 0;
	}
	else if(device_option == 2){
		cpu_test = 0;
		gpu_test = 1;
	}
	else{
		std::cout << "Device_option Invalid. Choose from the following:" << std::endl
				  << "  1. CPU" << std::endl
				  << "  2. GPU" << std::endl
				  << "Your device_option = " << device_option << std::endl;
	}

	double timeSec_Predict = 0;
	double timeSec_Eval = 0;
	double timeSec_MGS = 0;
	double timeSec = 0;

	CT* x_cpu;
	CT* x_cpu_target;

	if(cpu_test == 1){
		clock_t begin = clock();
		bool success = path_tracker(workspace_cpu, cpu_inst_hom, path_parameter,\
				 	 	 	 	 	 timeSec_Predict, timeSec_Eval, timeSec_MGS);
		clock_t end = clock();
		timeSec += (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
		x_cpu_target = workspace_cpu.x_last;
	}

	if(gpu_test == 1){
		bool success = GPU_Path(cpu_inst_hom, path_parameter, sol0, t, x_cpu_target);
	}

	// Path Tracking Reverse Test
	if(cpu_test == 1){
		t = CT(0.0,0.0);
		workspace_cpu.init_x_t_idx();
		workspace_cpu.update_x_t(x_cpu_target, t);
		bool success = path_tracker(workspace_cpu, cpu_inst_hom, path_parameter,\
					 	 	 	 	timeSec_Predict, timeSec_Eval, timeSec_MGS, 1);

		x_cpu = workspace_cpu.x_last;
	}

	if(gpu_test == 1){
		bool success = GPU_Path(cpu_inst_hom, path_parameter, x_cpu_target, t, x_cpu, 1);
	}

	//delete[] x_cpu;*/

	/*for(int i=0; i<cpu_inst_hom.dim; i++) {
		std::cout << i << " " << x_cpu[i];
	}*/

	std::cout << "--------- Start solution vs Reverse solution ----------" << std::endl;
	T1 err = err_check_workspace(sol0, x_cpu, cpu_inst_hom.dim);

	cout << "Path CPU Predict   Time: "<< timeSec_Predict << endl;
	cout << "Path CPU Eval      Time: "<< timeSec_Eval << endl;
	cout << "Path CPU MGS       Time: "<< timeSec_MGS << endl;

	return err;
}



int witness_set_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* x_cpu, CT t, int device_option) {
	std::cout << "--------- Witness Set Test ----------" << std::endl;

	bool cpu_test = 0;
	bool gpu_test = 0;
	if(device_option == 1){
			cpu_test = 1;
			gpu_test = 0;
	}
	else if(device_option == 2){
		cpu_test = 0;
		gpu_test = 1;
	}
	else{
		std::cout << "Device_option Invalid. Choose from the following:" << std::endl
				  << "  1. CPU" << std::endl
				  << "  2. GPU" << std::endl
				  << "Your device_option = " << device_option << std::endl;
	}


	CT* x_new = NULL;

	SolSet witness_set_start(cpu_inst_hom.dim);
	SolSet witness_set_target(cpu_inst_hom.dim);

	witness_set_start.add(x_cpu);
	std::cout << "Witness Set1" << std::endl;
	witness_set_start.print_short();

	double timeSec_Predict = 0;
	double timeSec_Eval = 0;
	double timeSec_MGS = 0;
	double timeSec = 0;

	for(int i=0; i<20; i++){

		clock_t begin = clock();

		bool success = 0;
		int alpha_round = 0;
		while(success==0){
			t = CT(0.0,0.0);
			workspace_cpu.init_x_t_idx();
			workspace_cpu.update_x_t(x_cpu, t);
			cpu_inst_hom.update_alpha();
			if(cpu_test == 1){
				success = path_tracker(workspace_cpu, cpu_inst_hom, path_parameter,\
						 timeSec_Predict, timeSec_Eval, timeSec_MGS);
			}
			if(gpu_test == 1){
				success = GPU_Path(cpu_inst_hom, path_parameter, x_cpu, t, x_new);
			}
			alpha_round++;
			if(success == 0 && alpha_round > 5){
				break;
			}
		}

		if(success == 0 && alpha_round > 5){
			std::cout << "Path Fail! Start -> Target" << std::endl \
					  << "alpha round = " << alpha_round << std::endl \
					  << "path  round = " << i << std::endl;
			break;
		}

		clock_t end = clock();
		timeSec += (end - begin) / static_cast<double>( CLOCKS_PER_SEC );

		if(cpu_test == 1){
			workspace_cpu.copy_x_last(x_cpu);
		}

		if(gpu_test == 1){
			delete[] x_cpu;
			x_cpu = x_new;
		}

		witness_set_target.add(x_cpu);

		std::cout << "Witness Set Target" << std::endl;
		witness_set_target.print_short();

		// Path Tracking Reverse Test
		success = 0;
		alpha_round = 0;
		while(success==0){
			t = CT(0.0,0.0);
			workspace_cpu.init_x_t_idx();
			workspace_cpu.update_x_t(x_cpu, t);
			cpu_inst_hom.update_alpha();
			//success = path_tracker(workspace_cpu, cpu_inst_hom, path_parameter,\
			//			 timeSec_Predict, timeSec_Eval, timeSec_MGS, 1);
			if(cpu_test == 1){
				success = path_tracker(workspace_cpu, cpu_inst_hom, path_parameter,\
						 timeSec_Predict, timeSec_Eval, timeSec_MGS, 1);
			}
			if(gpu_test == 1){
				success = GPU_Path(cpu_inst_hom, path_parameter, x_cpu, t, x_new, 1);
			}
			alpha_round++;
			if(success == 0 && alpha_round > 5){
				break;
			}
		}

		if(success == 0 && alpha_round > 5){
			std::cout << "Path Fail! Target -> Start" << std::endl \
					  << "alpha round = " << alpha_round << std::endl \
					  << "path  round = " << i << std::endl;
			break;
		}

		if(cpu_test == 1){
			workspace_cpu.copy_x_last(x_cpu);
		}

		if(gpu_test == 1){
			delete[] x_cpu;
			x_cpu = x_new;
		}

		witness_set_start.add(x_cpu);

		std::cout << "Witness Set Start" << std::endl;
		witness_set_start.print_short();
	}
	std::cout << "Witness Set Start" << std::endl;
	witness_set_start.print_short();
	std::cout << "Witness Set Target" << std::endl;
	witness_set_target.print_short();

	/*for(int i=0; i<cpu_inst_hom.dim; i++) {
		std::cout << i << " " << x_cpu[i];
	}*/

	/*CT* x_gpu = GPU_Path(cpu_inst_hom, sol0, t, n_predictor, max_it, err_max_delta_x, max_step);
	 cout << "Path CPU Test      Time: "<< timeSec << endl;
	 std::cout << "--------- Path Tracking Error CPU vs GPU ----------" << std::endl;
	 T1 err = err_check_workspace(x_cpu, x_gpu, cpu_inst_hom.dim);
	 std::cout << " x_cpu[0] = " << x_cpu[0];
	 std::cout << " x_gpu[0] = " << x_gpu[0];
	 free(x_gpu);*/

	//T1 err = err_check_workspace(sol0, x_cpu, cpu_inst_hom.dim);

	cout << "Path CPU Predict   Time: "<< timeSec_Predict << endl;
	cout << "Path CPU Eval      Time: "<< timeSec_Eval << endl;
	cout << "Path CPU MGS       Time: "<< timeSec_MGS << endl;

	return witness_set_start.n;
}

bool path_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, CT*& sol_new, CT t, PolySys& Target_Sys, int device_option) {
	std::cout << "--------- Path Tracking Test ----------" << std::endl;

	double timeSec_Path_CPU = 0;
	double timeMS_Path_CPU = 0;
	double timeSec_Predict_CPU = 0;
	double timeSec_Eval_CPU = 0;
	double timeSec_MGS_CPU = 0;
	CT* x_cpu = NULL;
	CT* x_gpu = NULL;

	bool cpu_test = 0;
	bool gpu_test = 0;

	bool cpu_success = 0;
	bool gpu_success = 0;

	T1 err = 0;

	if(device_option == 0){
		cpu_test = 1;
		gpu_test = 1;
	}
	else if(device_option == 1){
		cpu_test = 1;
		gpu_test = 0;
	}
	else if(device_option == 2){
		cpu_test = 0;
		gpu_test = 1;
	}
	else{
		std::cout << "Device_option Invalid. Choose from the following:" << std::endl
				  << "  0. CPU and GPU" << std::endl
				  << "  1. CPU" << std::endl
				  << "  2. GPU" << std::endl
				  << "Your device_option = " << device_option << std::endl;
	}

	if(cpu_test == 1){
		struct timeval start, end;
		long seconds, useconds;
		gettimeofday(&start, NULL);
		cpu_success = path_tracker(workspace_cpu, cpu_inst_hom, path_parameter,\
		             timeSec_Predict_CPU, timeSec_Eval_CPU, timeSec_MGS_CPU);
		x_cpu = workspace_cpu.x_last;
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		timeMS_Path_CPU = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		timeSec_Path_CPU = timeMS_Path_CPU/1000;
	}


	if(gpu_test == 1){
		gpu_success = GPU_Path(cpu_inst_hom, path_parameter, sol0, t, x_gpu);
		/*if(gpu_success == 1){
			// cyclic solution 0 pieri solution 2/3
			string file_name = sol_filename(cpu_inst_hom.dim, 0);
			write_complex_array(file_name, x_gpu, cpu_inst_hom.dim);
		}*/
	}

	if(cpu_test == 1){
		// Print Time and Error Report
		cout << "Path CPU Path MS   Time: "<< timeMS_Path_CPU << endl;
		cout << "Path CPU Path      Time: "<< timeSec_Path_CPU << endl;
		cout << "Path CPU Predict   Time: "<< timeSec_Predict_CPU << endl;
		cout << "Path CPU Eval      Time: "<< timeSec_Eval_CPU << endl;
		cout << "Path CPU MGS       Time: "<< timeSec_MGS_CPU << endl;
		cpu_inst_hom.timeSec_Path_CPU = timeSec_Path_CPU;
	}

	if(cpu_test == 1 && gpu_test == 1){
		std::cout << "--------- Path Tracking Error CPU vs GPU ----------" << std::endl;
		err = err_check_workspace(x_cpu, x_gpu, cpu_inst_hom.dim);
	}

	if(cpu_test == 1){
		std::cout << "CPU Solution" << std::endl;
		T1 max_x = 0;
		for(int i=0; i<cpu_inst_hom.dim; i++){
			//std::cout << i << "  " << x_cpu[i];
			if(abs(x_cpu[i].real) > max_x){
				max_x = abs(x_cpu[i].real);
			}
			if(abs(x_cpu[i].imag) > max_x){
				max_x = abs(x_cpu[i].imag);
			}
		}
		std::cout << "Max abs(x): " << max_x << std::endl;
	}

	if(gpu_test == 1){
		std::cout << "GPU Solution Max Abs ";
		T1 max_x = 0;
		for(int i=0; i<cpu_inst_hom.dim; i++){
			//std::cout << i << "  " << x_gpu[i];
			if(abs(x_gpu[i].real) > max_x){
				max_x = abs(x_gpu[i].real);
			}
			if(abs(x_gpu[i].imag) > max_x){
				max_x = abs(x_gpu[i].imag);
			}
		}
		std::cout << max_x << std::endl;
	}

	if(cpu_test == 1){
		std::cout << "CPU Path: ";
		if(cpu_success == 1){
			std::cout << "Success!" << std::endl;
			std::cout << "CPU Residual Check: ";
			int n_eq = cpu_inst_hom.n_eq;
			int dim = cpu_inst_hom.dim;
			CT* workspace_classic = new CT[n_eq*(dim+1)];
			CT* f_val = workspace_classic;
			CT** deri_val = new CT*[n_eq];
			CT* tmp_workspace = workspace_classic + n_eq;
			for(int i=0; i<n_eq; i++) {
				deri_val[i] = tmp_workspace;
				tmp_workspace += dim;
			}
			Target_Sys.eval(x_cpu, f_val, deri_val);
			T1 max_residual = 0;
			for(int i=0; i<n_eq; i++){
				//std::cout << i << " " << f_val[i];
				if(abs(f_val[i].real) > max_residual){
					max_residual = abs(f_val[i].real);
				}
				if(abs(f_val[i].imag) > max_residual){
					max_residual = abs(f_val[i].imag);
				}
			}
			std::cout << max_residual << std::endl;
			delete[] deri_val;
			delete[] workspace_classic;
		}
		else{
			std::cout << "Fail!" << std::endl;
		}

		sol_new = x_cpu;
	}

	if(gpu_test == 1){
		std::cout << "GPU Path: ";
		if(gpu_success == 1){
			std::cout << "Success!" << std::endl;
			std::cout << "GPU Residual Check: ";
			int n_eq = cpu_inst_hom.n_eq;
			int dim = cpu_inst_hom.dim;
			CT* workspace_classic = new CT[n_eq*(dim+1)];
			CT* f_val = workspace_classic;
			CT** deri_val = new CT*[n_eq];
			CT* tmp_workspace = workspace_classic + n_eq;
			for(int i=0; i<n_eq; i++) {
				deri_val[i] = tmp_workspace;
				tmp_workspace += dim;
			}
			Target_Sys.eval(x_gpu, f_val, deri_val);
			T1 max_residual = 0;
			for(int i=0; i<n_eq; i++){
				//std::cout << i << " " << f_val[i];
				if(abs(f_val[i].real) > max_residual){
					max_residual = abs(f_val[i].real);
				}
				if(abs(f_val[i].imag) > max_residual){
					max_residual = abs(f_val[i].imag);
				}
			}
			std::cout << max_residual << std::endl;
			delete[] deri_val;
			delete[] workspace_classic;
		}
		else{
			std::cout << "Fail!" << std::endl;
		}

		sol_new = x_gpu;
	}

	// if cpu or gpu tested, it has to be a success
	// no test or success


	cpu_inst_hom.success_CPU = cpu_success;
	cpu_inst_hom.success_GPU = gpu_success;

	bool success = (!cpu_test||cpu_success) && (!gpu_test||gpu_success);

	return success;
}

T1 predict_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, CT t) {
	std::cout << "--------- Predict Test ----------" << std::endl;
	workspace_cpu.init_x_t_predict_test();
	workspace_cpu.update_t_value(t);
	predictor_newton(workspace_cpu.x_array, workspace_cpu.t_array,
	workspace_cpu.x_t_idx, cpu_inst_hom.n_predictor, cpu_inst_hom.dim);
	CT* x_gpu;
	GPU_Predict(cpu_inst_hom, x_gpu, cpu_inst_hom.n_predictor, t);

	std::cout << "--------- GPU Predictor Error----------" << std::endl;
	T1 err = err_check_workspace(workspace_cpu.x, x_gpu, cpu_inst_hom.dim);
	std::cout << "x_cpu[0] = " << workspace_cpu.x[0];
	std::cout << "x_gpu[0] = " << x_gpu[0];
	free(x_gpu);
	return err;
}

T1 newton_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, Parameter path_parameter, CT* sol0, CT t) {
	std::cout << "--------- Newton Test ----------" << std::endl;

	double timeSec_Eval = 0;
	double timeSec_MGS = 0;

	struct timeval start, end;
	long seconds, useconds;
	double timeMS_cpu;
	gettimeofday(&start, NULL);

	bool success_cpu = CPU_Newton(workspace_cpu, cpu_inst_hom, path_parameter, \
			                      timeSec_Eval, timeSec_MGS);
	gettimeofday(&end, NULL);
	seconds  = end.tv_sec  - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	timeMS_cpu = ((seconds) * 1000 + useconds/1000.0) + 0.5;

	CT* x_gpu;
	bool success_gpu = GPU_Newton(cpu_inst_hom, path_parameter, sol0, t, x_gpu);
	cout << "Path CPU Newton    Time: "<< timeMS_cpu << endl;
	cout << "Path CPU Eval      Time: "<< timeSec_Eval << endl;
	cout << "Path CPU MGS       Time: "<< timeSec_MGS << endl;

	std::cout << "success_cpu = " << success_cpu << std::endl;
	std::cout << "success_gpu = " << success_gpu << std::endl;

	std::cout << "--------- Newton Error Check CPU VS GPU----------" << std::endl;
	T1 err = err_check_workspace(workspace_cpu.x, x_gpu, cpu_inst_hom.dim);
	std::cout << " x_cpu[0] = " << workspace_cpu.x[0];
	std::cout << " x_gpu[0] = " << x_gpu[0];

	delete[] x_gpu;
	x_gpu = NULL;

	return err;
}



T1 err_check_r(CT** CPU_R, CT* GPU_R, int dim, int right_hand_side=1) {
	T1 err(0);
	T1 err_bond(1E-10);
	int err_count = 0;
	int n_err_print = 10;

	int rows;
	int cols;

	if(right_hand_side == 1){
		rows = dim+1;
		cols = dim+1;
	}
	else{
		rows = dim;
		cols = dim;
	}

	for(int j=0; j<cols; j++){
		for(int i=0; i<=j; i++){
			if(i==dim && j == dim && right_hand_side == 1){
				break;
			}
			int tmp_idx =(dim+1)*(dim+2)/2 -(j+2)*(j+1)/2 + i;
			CT tmp_cpu = CPU_R[i][j];
			CT tmp_gpu = GPU_R[tmp_idx];

			//std::cout << i << " " << j << " " << GPU_R[tmp_idx];
			//std::cout  << i << " " << j << " " << CPU_R[i][j];

			T1 tmp_err;
			if(tmp_cpu.real == tmp_cpu.real && tmp_gpu.real == tmp_gpu.real){
				tmp_err = abs(tmp_cpu.real - tmp_gpu.real);
			}
			else{
				tmp_err = max(abs(tmp_cpu.real), abs(tmp_gpu.real));
			}
			if(tmp_err > err_bond) {
				err_count++;
				if(err_count < n_err_print) {
					std::cout << "d" << i << " " << j << " = " << tmp_cpu
					<< "     " << tmp_gpu;
				}
			}
			if(tmp_err > err) {
				err = tmp_err;
			}

			if(tmp_cpu.imag == tmp_cpu.imag && tmp_gpu.imag == tmp_gpu.imag){
				tmp_err = abs(tmp_cpu.imag - tmp_gpu.imag);
			}
			else{
				tmp_err = max(abs(tmp_cpu.imag), abs(tmp_gpu.imag));
			}
			if(tmp_err > err_bond) {
				err_count++;
				if(err_count < n_err_print) {
					std::cout << "d" << i << " " << j << " = " << tmp_cpu
					<< "     " << tmp_gpu;
				}
			}

			if(tmp_err > err) {
				err = tmp_err;
			}
		}
	}

	/*for(int i=0; i<n_workspace_size; i++) {
		T1 tmp_err;
		if(workspace1[i].real == workspace1[i].real && workspace2[i].real == workspace2[i].real){
			tmp_err = abs(workspace1[i].real - workspace2[i].real);
		}
		else{
			tmp_err = max(abs(workspace1[i].real), abs(workspace2[i].real));
		}
		if(tmp_err > err_bond) {
			err_count++;
			if(err_count < n_err_print) {
				std::cout << "d" << i << " = " << workspace1[i]
				<< "     " << workspace2[i];
			}
		}
		if(tmp_err > err) {
			err = tmp_err;
		}

		if(workspace1[i].imag == workspace1[i].imag && workspace2[i].imag == workspace2[i].imag){
			tmp_err = abs(workspace1[i].imag - workspace2[i].imag);
		}
		else{
			tmp_err = max(abs(workspace1[i].imag), abs(workspace2[i].imag));
		}
		if(tmp_err > err_bond) {
			err_count++;
			if(err_count < n_err_print) {
				std::cout << "d" << i << " = " << workspace1[i]
				<< "     " << workspace2[i];
			}
		}

		if(tmp_err > err) {
			err = tmp_err;
		}
	}*/

	if(err_count > 0) {
		std::cout << "n_err = " << err_count << std::endl;
	}
	std::cout << "err = " << std::scientific << err << std::endl;
	return err;
}

T1 mgs_test_any(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, int device_option) {
	std::cout << "--------------- Modified Gram-Smith Test ----------"
			<< std::endl;

	bool cpu_test = 0;
	bool gpu_test = 0;
	if(device_option == 0){
		cpu_test = 1;
		gpu_test = 1;
		std::cout << "CPU + GPU Testing..." << std::endl;
	}
	else if(device_option == 1){
		cpu_test = 1;
		gpu_test = 0;
		std::cout << "CPU Testing..." << std::endl;
	}
	else if(device_option == 2){
		cpu_test = 0;
		gpu_test = 1;
		std::cout << "GPU Testing..." << std::endl;
	}

	CT* x = workspace_cpu.x;
	CT t = *(workspace_cpu.t);
	CT** V = (workspace_cpu.V);
	CT** R = (workspace_cpu.R);
	CT* sol_cpu = (workspace_cpu.sol);

	int dim = cpu_inst_hom.dim;
	int n_eq = cpu_inst_hom.n_eq; // to be removed

	std::cout << "cpu_inst_hom.dim  = " << cpu_inst_hom.dim << std::endl;
	std::cout << "cpu_inst_hom.n_eq = " << cpu_inst_hom.n_eq << std::endl;

	//cpu_inst_hom.eval(workspace_cpu, x, t);

	CT* sol_gpu = NULL;
	CT* matrix_gpu_q = NULL;
	CT* matrix_gpu_r = NULL;
	CT* tmp_right = new CT[n_eq];

	srand(time(NULL));
	//srand(1);
	for(int i=0; i<dim+1; i++){
		for(int j=0; j<n_eq; j++) {
			/*if(i>=j){
				workspace_cpu.matrix[i*n_eq+j] = CT(1+i-j,1);
			}

			//if(i<=j){
			//	workspace_cpu.matrix[i*n_eq+j] = CT(1-i+j,0);
			//}

			//if(i==j){
				//workspace_cpu.matrix[i*n_eq+j] = CT((i+j)%5+1,0);
			//}
			else{
				//workspace_cpu.matrix[i*n_eq+j] = CT(0.0,0);
				workspace_cpu.matrix[i*n_eq+j] = CT(1,0);
			}*/

			double temp = rand()/((double) RAND_MAX)*2*M_PI;
			workspace_cpu.matrix[i*n_eq+j] = CT(cos(temp),sin(temp));

			//workspace_cpu.matrix[i*n_eq+j] = CT((i+j)%5+1,0);
			//std::cout << i << " " << j << " " << workspace_cpu.matrix[i*n_eq+j];
			//workspace_cpu.matrix[i*n_eq+j].imag = 0;
		}
	}

	// Save matrix for right hand side check
	CT* tmp_matrix = new CT[n_eq*(dim+1)];
	for (int i = 0; i < n_eq * (dim + 1); i++) {
		tmp_matrix[i] = workspace_cpu.matrix[i];
	}

	//workspace_cpu.print_result(dim,n_eq);

	if(gpu_test == 1){
	// GPU MGS
		GPU_MGS_Large(cpu_inst_hom, sol_gpu, matrix_gpu_q, matrix_gpu_r,\
					  cpu_inst_hom.n_predictor, workspace_cpu.matrix);
	}
	// CPU MGS

	if(cpu_test == 1){
		struct timeval start, end;
		long seconds, useconds;
		gettimeofday(&start, NULL);
		CPU_mgs2qrls(V, R, sol_cpu, n_eq, dim + 1);
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		double timeMS_MGS_CPU = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		double timeSec_MGS_CPU = timeMS_MGS_CPU/1000;
		std::cout << "CPU Time MS  = " << timeMS_MGS_CPU << std::endl;
		std::cout << "CPU Time Sec = " << timeSec_MGS_CPU << std::endl;
	}

	T1 err1 = 0;
	T1 err2 = 0;
	T1 err3 = 0;


	if(cpu_test == 1 && gpu_test == 1){
		std::cout << "----------- Solution Check CPU vs GPU-------------"
				<< std::endl;
		err2 = err_check_workspace(sol_cpu, sol_gpu, cpu_inst_hom.dim);

		cout << "         sol[0] = " << sol_cpu[0];
		cout << "     sol_gpu[0] = " << sol_gpu[0];
		cout << "         sol[1] = " << sol_cpu[1];
		cout << "     sol_gpu[1] = " << sol_gpu[1];
		cout << "       v_cpu[0] = " << V[0][0];
		cout << "       v_gpu[0] = " << matrix_gpu_q[0];
		cout << "       v_cpu[1] = " << V[0][1];
		cout << "       v_gpu[1] = " << matrix_gpu_q[1];

		std::cout << "----------- Q Check CPU vs GPU-------------"
				<< (cpu_inst_hom.dim + 1) * cpu_inst_hom.n_eq << std::endl;
		err3 = err_check_workspace_matrix(workspace_cpu.matrix, matrix_gpu_q,\
									  cpu_inst_hom.n_eq,(cpu_inst_hom.dim+1));

		std::cout << "----------- R Check CPU vs GPU -------------" << std::endl;
		err_check_r(R, matrix_gpu_r, cpu_inst_hom.dim, 0);
	}


	std::cout << "----------- Right hand side check -------------" << std::endl;
	if(cpu_test == 1){
		std::cout << "CPU ";
		for (int i = 0; i < n_eq; i++) {
			tmp_right[i] = CT(0.0,0);
			for(int j=0; j<dim; j++) {
				tmp_right[i] += sol_cpu[j]*tmp_matrix[j*n_eq+i];
			}
		}
		err1 = err_check_workspace(tmp_right, tmp_matrix + dim * n_eq,
				cpu_inst_hom.n_eq);
	}

	if(gpu_test == 1){
		std::cout << "GPU ";
		for (int i = 0; i < n_eq; i++) {
			tmp_right[i] = CT(0.0,0);
			for(int j=0; j<dim; j++) {
				tmp_right[i] += sol_gpu[j]*tmp_matrix[j*n_eq+i];
			}
		}
		err1 = err_check_workspace(tmp_right, tmp_matrix + dim * n_eq,
				cpu_inst_hom.n_eq);
	}

	delete[] tmp_matrix;
	delete[] tmp_right;

	free(sol_gpu);
	free(matrix_gpu_q);
	free(matrix_gpu_r);
	return max(err1, err2);
}

T1 mgs_test_large(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom) {
	std::cout << "--------------- Modified Gram-Smith Test ----------"
			<< std::endl;
	CT* x = workspace_cpu.x;
	CT t = *(workspace_cpu.t);
	CT** V = (workspace_cpu.V);
	CT** R = (workspace_cpu.R);
	CT* sol_cpu = (workspace_cpu.sol);
	int dim = cpu_inst_hom.dim;
	int n_eq = cpu_inst_hom.n_eq; // to be removed

	cpu_inst_hom.eval(workspace_cpu, x, t);

	CT* sol_gpu = NULL;
	CT* matrix_gpu_q = NULL;
	CT* matrix_gpu_r = NULL;

	/*for(int i=0; i<dim+1; i++){
		for(int j=0; j<n_eq; j++) {
			if(i>=j){
				workspace_cpu.matrix[i*n_eq+j] = CT(1+i-j,1);
			}

			//if(i<=j){
			//	workspace_cpu.matrix[i*n_eq+j] = CT(1-i+j,0);
			//}

			//if(i==j){
				//workspace_cpu.matrix[i*n_eq+j] = CT((i+j)%5+1,0);
			//}
			else{
				workspace_cpu.matrix[i*n_eq+j] = CT(0.0,0);
				//workspace_cpu.matrix[i*n_eq+j] = CT(1,0);
			}
			//workspace_cpu.matrix[i*n_eq+j] = CT((i+j)%5+1,0);
			std::cout << i << " " << j << " " << workspace_cpu.matrix[i*n_eq+j];
			//workspace_cpu.matrix[i*n_eq+j].imag = 0;
		}
	}*/
	// Save matrix for right hand side check
	CT* tmp_matrix = new CT[n_eq*(dim+1)];
	for (int i = 0; i < n_eq * (dim + 1); i++) {
		tmp_matrix[i] = workspace_cpu.matrix[i];
	}

	//workspace_cpu.print_result(dim,n_eq);

	// GPU MGS
	GPU_MGS_Large(cpu_inst_hom, sol_gpu, matrix_gpu_q, matrix_gpu_r,\
			      cpu_inst_hom.n_predictor, workspace_cpu.matrix);
	// CPU MGS
	CPU_mgs2qrls(V, R, sol_cpu, n_eq, dim + 1);

	std::cout << "----------- Right hand side check -------------" << std::endl;
	CT* tmp_right = new CT[n_eq];
	for (int i = 0; i < n_eq; i++) {
		tmp_right[i] = CT(0.0,0);
		for(int j=0; j<dim; j++) {
			tmp_right[i] += sol_cpu[j]*tmp_matrix[j*n_eq+i];
		}
	}
	T1 err1 = err_check_workspace(tmp_right, tmp_matrix + dim * n_eq,
			cpu_inst_hom.n_eq);

	delete[] tmp_matrix;
	delete[] tmp_right;

	T1 err2 = 0;
	T1 err3 = 0;

	std::cout << "----------- Solution Check CPU vs GPU-------------"
			<< std::endl;
	err2 = err_check_workspace(sol_cpu, sol_gpu, cpu_inst_hom.dim);

	cout << "         sol[0] = " << sol_cpu[0];
	cout << "     sol_gpu[0] = " << sol_gpu[0];
	cout << "       v_cpu[0] = " << V[0][0];
	cout << "       v_gpu[0] = " << matrix_gpu_q[0];
	cout << "       v_cpu[1] = " << V[0][1];
	cout << "       v_gpu[1] = " << matrix_gpu_q[1];

	std::cout << "----------- Q Check CPU vs GPU-------------"
			<< (cpu_inst_hom.dim + 1) * cpu_inst_hom.n_eq << std::endl;
	err3 = err_check_workspace_matrix(workspace_cpu.matrix, matrix_gpu_q,\
			                      cpu_inst_hom.n_eq,(cpu_inst_hom.dim+1));

	std::cout << "----------- R Check CPU vs GPU -------------" << std::endl;
	err_check_r(R, matrix_gpu_r, cpu_inst_hom.dim, 0);

	free(sol_gpu);
	free(matrix_gpu_q);
	free(matrix_gpu_r);
	return max(err1, err2);
}

T1 mgs_test(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom) {
	std::cout << "--------------- Modified Gram-Smith Test ----------"
			<< std::endl;
	CT* x = workspace_cpu.x;
	CT t = *(workspace_cpu.t);
	CT** V = (workspace_cpu.V);
	CT** R = (workspace_cpu.R);
	CT* sol_cpu = (workspace_cpu.sol);
	int dim = cpu_inst_hom.dim;
	int n_eq = cpu_inst_hom.n_eq; // to be removed

	cpu_inst_hom.eval(workspace_cpu, x, t);

	CT* sol_gpu = NULL;
	CT* matrix_gpu_q = NULL;
	CT* matrix_gpu_r = NULL;

	// Save matrix for right hand side check
	CT* tmp_matrix = new CT[n_eq*(dim+1)];
	for (int i = 0; i < n_eq * (dim + 1); i++) {
		tmp_matrix[i] = workspace_cpu.matrix[i];
	}

	//workspace_cpu.print_result(dim,n_eq);

	// GPU MGS
	GPU_MGS(cpu_inst_hom, sol_gpu, matrix_gpu_q, matrix_gpu_r,\
			cpu_inst_hom.n_predictor, workspace_cpu.matrix);
	// CPU MGS
	CPU_mgs2qrls(V, R, sol_cpu, n_eq, dim + 1);

	std::cout << "----------- Right hand side check -------------" << std::endl;
	CT* tmp_right = new CT[n_eq];
	for (int i = 0; i < n_eq; i++) {
		tmp_right[i] = CT(0.0,0);
		for(int j=0; j<dim; j++) {
			tmp_right[i] += sol_cpu[j]*tmp_matrix[j*n_eq+i];
		}
	}
	T1 err1 = err_check_workspace(tmp_right, tmp_matrix + dim * n_eq,
			                      cpu_inst_hom.n_eq);

	delete[] tmp_matrix;
	delete[] tmp_right;

	std::cout << "----------- Solution Check CPU vs GPU-------------"
			<< std::endl;
	T1 err2 = err_check_workspace(sol_cpu, sol_gpu, cpu_inst_hom.dim);

	std::cout << "----------- Q Check CPU vs GPU -------------"
			<< (cpu_inst_hom.dim + 1) * cpu_inst_hom.n_eq << std::endl;
	T1 err3 = err_check_workspace_matrix(workspace_cpu.matrix, matrix_gpu_q,\
			cpu_inst_hom.n_eq,(cpu_inst_hom.dim + 1));

	std::cout << "----------- R Check CPU vs GPU -------------" << std::endl;
	err_check_r(R, matrix_gpu_r, dim);

	cout << "         sol[0] = " << sol_cpu[0];
	cout << "     sol_gpu[0] = " << sol_gpu[0];
	cout << "       v_cpu[0] = " << V[0][0];
	cout << "       v_gpu[0] = " << matrix_gpu_q[0];
	cout << "       v_cpu[1] = " << V[0][1];
	cout << "       v_gpu[1] = " << matrix_gpu_q[1];

	free(sol_gpu);
	free(matrix_gpu_q);
	free(matrix_gpu_r);

	return max(err1, err2);
}

T1 err_check_workspace(const CT* workspace1, const CT* workspace2, int n_workspace_size, int n_err_print) {
	T1 err(0);
	T1 err_bond(1E-10);
	int err_count = 0;
	for(int i=0; i<n_workspace_size; i++) {
		bool err_print = 0;
		T1 tmp_err;
		if(workspace1[i].real == workspace1[i].real && workspace2[i].real == workspace2[i].real){
			tmp_err = abs(workspace1[i].real - workspace2[i].real);
		}
		else{
			tmp_err = max(abs(workspace1[i].real), abs(workspace2[i].real));
		}
		if(tmp_err > err_bond || workspace1[i].real != workspace1[i].real || workspace2[i].real != workspace2[i].real) {
			err_count++;
			if(err_count < n_err_print) {
				std::cout << "d" << i << " = " << workspace1[i]
				<< "     " << workspace2[i];
				err_print = 1;
			}
		}
		if(tmp_err > err) {
			err = tmp_err;
		}

		if(workspace1[i].imag == workspace1[i].imag && workspace2[i].imag == workspace2[i].imag){
			tmp_err = abs(workspace1[i].imag - workspace2[i].imag);
		}
		else{
			tmp_err = max(abs(workspace1[i].imag), abs(workspace2[i].imag));
		}
		if(tmp_err > err_bond || workspace1[i].imag != workspace1[i].imag || workspace2[i].imag != workspace2[i].imag) {
			err_count++;
			if((err_count < n_err_print) && (err_print == 0)) {
				std::cout << "d" << i << " = " << workspace1[i]
				<< "     " << workspace2[i];
			}
		}

		if(tmp_err > err) {
			err = tmp_err;
		}

	}
	if(err_count > 0) {
		std::cout << "n_err = " << err_count << std::endl;
	}
	std::cout << "err = " << std::scientific << err << std::endl;
	return err;
}

T1 err_check_workspace_matrix(const CT* workspace1, const CT* workspace2, int n_rows, int n_cols) {
	T1 err(0);
	T1 err_bond(1E-10);
	int err_count = 0;
	int n_err_print = 20;
	int i=0;
	for(int col=0; col<n_cols; col++){
		for(int row=0; row<n_rows; row++) {
			bool err_print = 0;
			T1 tmp_err;
			if(workspace1[i].real == workspace1[i].real && workspace2[i].real == workspace2[i].real){
				tmp_err = abs(workspace1[i].real - workspace2[i].real);
			}
			else{
				tmp_err = max(abs(workspace1[i].real), abs(workspace2[i].real));
			}
			if(tmp_err > err_bond || workspace1[i].real != workspace1[i].real || workspace2[i].real != workspace2[i].real) {
				err_count++;
				if(err_count < n_err_print) {
					std::cout << "col="<< col << " "<< "row="<< row << " " << workspace1[i]\
					                               << "              " << workspace2[i];
					err_print = 1;
				}
			}
			if(tmp_err > err) {
				err = tmp_err;
			}

			if(workspace1[i].imag == workspace1[i].imag && workspace2[i].imag == workspace2[i].imag){
				tmp_err = abs(workspace1[i].imag - workspace2[i].imag);
			}
			else{
				tmp_err = max(abs(workspace1[i].imag), abs(workspace2[i].imag));
			}
			if(tmp_err > err_bond || workspace1[i].imag != workspace1[i].imag || workspace2[i].imag != workspace2[i].imag) {
				err_count++;
				if((err_count < n_err_print) && (err_print == 0)) {
					std::cout << "col="<< col << " "<< "row="<< row << " " << workspace1[i]\
					                               << "              " << workspace2[i];
				}
			}

			if(tmp_err > err) {
				err = tmp_err;
			}
			i++;
		}
	}
	if(err_count > 0) {
		std::cout << "n_err = " << err_count << std::endl;
	}
	std::cout << "err = " << std::scientific << err << std::endl;
	return err;
}

void err_check_class_workspace(CT** deri_val, CT* f_val, CT* matrix, int n_eq, int dim) {
	T1 err(0);
	T1 err_bond(1E-10);
	for(int i=0; i<n_eq; i++) {
		for(int j=0; j<dim; j++) {
			T1 tmp_err = abs(deri_val[i][j].real - matrix[j*n_eq + i].real);
			if(tmp_err > err_bond) {
				std::cout << "d" << i << " " << j << " = " << deri_val[i][j]
				<< "       " << matrix[j*n_eq + i];
			}
			if(tmp_err > err) {
				err = tmp_err;
			}
			tmp_err = abs(deri_val[i][j].imag - matrix[j*n_eq + i].imag);
			if(tmp_err > err_bond) {
				err = tmp_err;
				std::cout << "d" << i << " " << j << " = " << deri_val[i][j]
				<< "       " << matrix[j*n_eq + i];
			}
			if(tmp_err > err) {
				err = tmp_err;
			}

		}
		T1 tmp_err = abs(f_val[i].real - matrix[dim*n_eq + i].real);
		if(tmp_err > err_bond) {
			std::cout << "f" << i << " = " << f_val[i]
			<< "     " << matrix[dim*n_eq + i];
		}
		if(tmp_err > err) {
			err = tmp_err;
		}
		tmp_err = f_val[i].imag - matrix[dim*n_eq + i].imag;
		if(tmp_err > err_bond) {
			err = tmp_err;
			std::cout << "f" << i << " = " << f_val[i]
			<< "    " << matrix[dim*n_eq + i];
		}
		if(tmp_err > err) {
			err = tmp_err;
		}
	}

	std::cout << "err = " << std::scientific << err << std::endl;
}

T1 eval_test(const CPUInstHom& cpu_inst_hom, CT* host_sol0, CT t, const CT* cpu_workspace, const CT* cpu_matrix)
{
	int n_coef = cpu_inst_hom.CPU_inst_hom_coef.n_coef;
	int n_workspace_size = n_coef + cpu_inst_hom.CPU_inst_hom_mon.mon_pos_size;
	CT* gpu_workspace;
	CT* gpu_matrix;

	GPU_Eval(cpu_inst_hom, host_sol0, t, gpu_workspace, gpu_matrix);

	int n_workspace = n_coef+cpu_inst_hom.CPU_inst_hom_mon.mon_pos_size;

	std::cout << "----- Coef Check CPU vs GPU ----" << std::endl;
	err_check_workspace(cpu_workspace, gpu_workspace, n_coef);

	std::cout << "----- Workspace Check CPU vs GPU ----" << std::endl;
	err_check_workspace(cpu_workspace, gpu_workspace, n_workspace);

	std::cout << "----- Jacobian and Fun Check CPU vs GPU ----" << std::endl;
	T1 err = err_check_workspace(cpu_matrix, gpu_matrix, cpu_inst_hom.n_eq*(cpu_inst_hom.dim+1));

	return err;
}

void generate_cyclic_system(PolySys& Target_Sys, PolySys& Start_Sys,
		PolySolSet& sol_set, int dim) {
	string* sys_string = string_cyclic(dim);
	for (int i = 0; i < dim; i++) {
		std::cout << sys_string[i] << std::endl;
	}

	VarDict pos_dict_target;
	Target_Sys.read(sys_string, dim, pos_dict_target);

	string x_name = "x";
	string* x_names = x_var(x_name, dim);
	Target_Sys.pos_var = x_names;

	Target_Sys.print();

	VarDict pos_dict;

	std::ostringstream cyclic_filename;
	cyclic_filename << "cyclic" << dim << ".start";

	ifstream myfile(cyclic_filename.str().c_str());

	Start_Sys.read_file(myfile, pos_dict);

	sol_set.init(myfile);
	//sol_set.print();

	string v_name = "z";
	string* v_names = x_var(v_name, dim);
	Start_Sys.pos_var = v_names;

	Start_Sys.print();
}

T1 eval_test_classic(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, CT* sol0, CT t, PolySys& Classic_Sys, int n_eq, int dim) {
	std::cout << "----- Class Evaluation ----" << std::endl;
	CT* workspace_classic = new CT[n_eq*(dim+1)];
	CT* f_val = workspace_classic;
	CT** deri_val = new CT*[n_eq];
	CT* tmp_workspace = workspace_classic + n_eq;
	for(int i=0; i<n_eq; i++) {
		deri_val[i] = tmp_workspace;
		tmp_workspace += dim;
	}

	clock_t begin_classic = clock();
	Classic_Sys.eval(sol0, f_val, deri_val);
	clock_t end_classic = clock();
	double timeSec_classic = (end_classic - begin_classic) / static_cast<double>( CLOCKS_PER_SEC );

	/*for(int i=0; i<n_eq; i++){
		for(int j=0; j<dim; j++){
		std::cout << i << " " << j << " " << deri_val[i][j];
		}
		std::cout << "f" << i << " = "<< f_val[i];
	}*/

	std::cout << "----- Instruction Evaluation ----" << std::endl;
	clock_t begin_cpu = clock();
	cpu_inst_hom.eval(workspace_cpu, sol0, t);
	clock_t end_cpu = clock();
	double timeSec_cpu = (end_cpu - begin_cpu) / static_cast<double>( CLOCKS_PER_SEC );

	// Check two CPU method
	std::cout << "----- Classic Evaluation Check ----" << std::endl;
	err_check_class_workspace(deri_val, f_val, workspace_cpu.matrix, n_eq, dim);

	clock_t begin_gpu = clock();
	T1 err = eval_test(cpu_inst_hom, sol0, t, workspace_cpu.all, workspace_cpu.matrix);
	clock_t end_gpu = clock();
	double timeSec_gpu = (end_gpu - begin_gpu) / static_cast<double>( CLOCKS_PER_SEC );

	delete[] workspace_classic;

	std::cout << "Classic Eval time " << timeSec_classic << std::endl;
	std::cout << "CPU     Eval time " << timeSec_cpu << std::endl;
	std::cout << "GPU     Eval time " << timeSec_gpu << std::endl;

	return err;
}
