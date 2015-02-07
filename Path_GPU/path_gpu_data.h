#ifndef GPU_DATA_H_
#define GPU_DATA_H_

#include "DefineType.h"
#include "gqd_qd_util.h"
#include "eval_host.h"


int get_NB(int n_job, int BS, int n_thread_per_job = 1);

dim3 get_grid(int NB, int n_sys=1);

dim3 get_grid(int n_job, int BS, int n_sys, int n_thread_per_job = 1);

class GPUWorkspace{
public:
	GT* all;
	GT* mon;
	GT* coef;
	GT* sum;

	int n;
	size_t size_all;

	GT* matrix; //jacobian and fun value
	int n_matrix;
	size_t size_matrix;

	// Array for predictor usage
	int dim;
	int n_eq;
	int n_array;
	int n_predictor;

	int x_t_idx;

	GT* x_array;
	GT* x;
	GT* x_last;
	size_t x_array_size;

	GT* t_array;
	GT* t;
	GT* t_last;
	GT* one_minor_t;
	size_t t_array_size;
	CT alpha;

	GT* V;
	GT* R;
	GT* sol;
	size_t V_size;
	size_t R_size;
	size_t sol_size;

	GT* f_val; //fun value
	GT* P;

	GPUWorkspace(int n, int n_coef, int n_constant, int n_eq, int dim, int n_predictor, CT alpha=CT(1,0)){
		this->n = n;
		this->dim = dim;
		this->n_eq = n_eq;
		size_all = n*sizeof(GT);
		cudaMalloc((void **)&all, size_all);
		coef = all;
		mon = coef + n_coef;
		sum = mon - n_constant;

		n_matrix = n_eq*(dim+1);
		size_matrix = n_matrix*sizeof(GT);
		cudaMalloc((void **)&matrix, size_matrix);
		f_val = matrix + n_eq*dim;

		init_x_t(dim, n_predictor);
		init_matrix(dim, n_eq);
		this->alpha = alpha;
	}

	~GPUWorkspace(){
		std::cout << "Delete GPUWorkspace" << std::endl;
		cudaFree(all);
		cudaFree(matrix);
		cudaFree(R);
		cudaFree(sol);
		cudaFree(x_array);
		cudaFree(t_array);
		cudaFree(one_minor_t);
	}

	void init_matrix(int dim, int n_eq);

	void init_V_value(CT* V_cpu);

	void init_x_t(int dim, int n_predictor);

	void update_x_t_idx();

	void update_t_value(CT cpu_t);


	void update_t_value_inverse(CT cpu_one_minor_t);

	void update_x_value(CT* cpu_sol0);

	void update_x_t_value(CT* cpu_sol0, CT cpu_t);

	void update_x_t(CT* cpu_sol0, CT cpu_t);

	CT* get_workspace();

	CT* get_matrix();

	CT* get_matrix_r();

	void print_matrix_r();

	CT* get_x();

	CT* get_x_array();

	void print_x();

	void print_x_array();

	CT* get_x_last();

	CT* get_sol();

	T1 sol_norm();

	void init_x_t_predict_test();
};

class GPUInst{
public:
	int n_sys;

	//Sol Instruction
	int dim;
	int n_eq;

	/**** workspace Instruction ***/
	int n_workspace;
	int n_constant;

	// Coef Instruction
	int n_coef;
	GT* coef;

	int coef_BS;
	dim3 coef_grid;

	int dim_BS;
	dim3 dim_grid;


	/**** Mon Instruction ****/
	// for leveled kernels
	int level;
	int* n_mon_level;
	// for single kernel
	int n_mon;
	int* mon_pos_start;
	unsigned short* mon_pos;

	int n_mon_global;

	dim3* mon_level_grid;
	int* n_mon_level_rest;
	dim3* mon_level_grid_rest;

	int mon_level0_BS;
	int mon_level_BS;

	int mon_global_BS;
	dim3 mon_global_grid;

	int n_mon_block;
	int BS_mon_block;
	int NB_mon_block;
	int* mon_pos_start_block;
	unsigned short* mon_pos_block;


	/**** Sum instruction ****/
	int n_sum; // size of sum_start
	int n_sum_levels;
	int* n_sum_level;
	dim3* sum_level_grid;
	int* n_sum_level_rest;
	dim3* sum_level_grid_rest;

	int* sum_pos_start;
	int* sum_pos;

	int sum_BS;
	dim3 sum_grid;

	int n_step_GPU;
	int n_point_GPU;
	int n_eval_GPU;
	int n_mgs_GPU;

	CT alpha;

	GPUInst(const CPUInstHom& cpu_inst, int n_sys){
		dim = cpu_inst.dim;
		n_eq = cpu_inst.n_eq;
		this->n_sys = n_sys;
		init_coef(cpu_inst.CPU_inst_hom_coef);
		init_mon(cpu_inst.CPU_inst_hom_mon, cpu_inst.CPU_inst_hom_block);
		if(MON_EVAL_METHOD == 3){
			init_sum(cpu_inst.CPU_inst_hom_sum_block);
		}
		else{
			init_sum(cpu_inst.CPU_inst_hom_sum);
		}
		init_workspace(cpu_inst);

		dim_BS = 32;
	    dim_grid = get_grid(dim,dim_BS,n_sys);
	    n_step_GPU = 0;
	    n_point_GPU = 0;
	    n_eval_GPU = 0;
	    n_mgs_GPU = 0;
	}

	~GPUInst(){
		cudaFree(coef);
		cudaFree(mon_pos_start);
		cudaFree(mon_pos);
		cudaFree(sum_pos_start);
		cudaFree(sum_pos);
	}

	void init_coef(const CPUInstHomCoef& cpu_inst_coef);

	void init_mon(const CPUInstHomMon& cpu_inst_mon, const CPUInstHomMonBlock& cpu_inst_mon_block);

	void init_sum(const CPUInstHomSumBlock& cpu_inst_sum);

	void init_sum(const CPUInstHomSum& cpu_inst_sum);

	void init_workspace(const CPUInstHom& cpu_inst);
};

#endif /* GPU_DATA_H_ */

