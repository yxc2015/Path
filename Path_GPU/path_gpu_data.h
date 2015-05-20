#ifndef GPU_DATA_H_
#define GPU_DATA_H_

#include "DefineType.h"
#include "gqd_qd_util.h"
#include "eval_host.h"


int get_NB(int n_job, int BS, int n_thread_per_job = 1);

dim3 get_grid(int NB, int n_path=1);

dim3 get_grid(int n_job, int BS, int n_path, int n_thread_per_job = 1);

class GPUWorkspace{
public:

	// Array for predictor usage
	int dim;
	int n_eq;
	int n_array;
	int n_predictor;
	int mon_pos_size;
	int n_constant;
	int n_coef;
	int n_path;
	int n_path_continuous;

	CT alpha;
	GT* alpha_gpu;

	GT* all;
	GT* mon;
	GT* coef;
	GT* sum;

	int workspace_size;
	int n;
	size_t size_all;

	GT* matrix; //jacobian and fun value
	int n_matrix;
	int n_matrix_R;
	size_t size_matrix;

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

	GT* V;
	GT* R;
	GT* sol;
	size_t V_size;
	size_t R_size;
	size_t sol_size;

	GT* f_val; //fun value
	GT* P;

	// int arrays
	int* int_arrays;
	int* x_t_idx_mult;
	int* n_point_mult;
	int* path_idx;
	int* path_success;
	int* n_success;
	int* end_range;
	int* newton_success;

	double* double_arrays;
	double* max_delta_x_gpu;
	double* r_max_delta_x_gpu;
	double* max_f_val_gpu;
	double* max_f_val_last_gpu;
	double* r_max_f_val_gpu;
	double* max_x_gpu;

	size_t size_GT_arrays;
	int n_GT_arrays;
	GT* GT_arrays;
	GT* x_mult;
	GT* t_mult;
	GT* t_last_mult;
	GT* delta_t_mult;
	GT* newton_t_mult;
	GT* one_minor_t_mult;
	GT* coef_mult;
	GT* sum_mult;
	GT* mon_mult;
	GT* matrix_mult;
	GT* f_val_mult;
	GT* matrix_mult_horizontal;
	GT* V_mult;
	GT* R_mult;
	GT* sol_mult;
	GT* x_array_mult;
	GT* t_array_mult;

	GT* workspace_eq;

	int* newton_success_host;
	int* path_idx_host;
	int* path_success_host;

	double* max_delta_x_host;
	double* r_max_delta_x_host;
	double* max_f_val_host;
	double* r_max_f_val_host;
	double* max_x_host;

	GPUWorkspace(int n, int mon_pos_size, int n_coef, int n_constant, int n_eq, int dim, int n_predictor, CT alpha=CT(1,0), int n_path=1){
		this->mon_pos_size = mon_pos_size;
		this->dim = dim;
		this->n_eq = n_eq;
		this->n_coef = n_coef;
		this->n_constant = n_constant;
		this->n_path = n_path;
		this->n_path_continuous = n_path;
		this->n_predictor = n_predictor;
		this->dim = dim;
		n_array = n_predictor + 1;

		this->alpha = alpha;
		cudaMalloc((void **)&alpha_gpu, sizeof(GT));
		GT* alpha_gpu_host = (GT *)malloc(sizeof(GT));
		comp1_qd2gqd(&alpha, alpha_gpu_host);
		cudaMemcpy(alpha_gpu, alpha_gpu_host, sizeof(GT), \
					cudaMemcpyHostToDevice);

		this->n = n;

		size_all = n*sizeof(GT);
		//cudaMalloc((void **)&all, size_all);

		workspace_size = n;

		n_matrix = n_eq*(dim+1);
		size_matrix = n_matrix*sizeof(GT);
		workspace_size += n_matrix;

		V_size = size_matrix;
		R_size = (dim + 1)*(dim + 1)*sizeof(GT);
		workspace_size += (dim + 1)*(dim + 1);

		sol_size = dim*sizeof(GT);
		std::cout << "dim = " << dim << " " << sol_size << std::endl;
		workspace_size += dim;

		int rows = n_eq;
		int cols = dim+1;
		int row_block = (rows-1)/matrix_block_row+1;
		workspace_size += row_block*matrix_block_pivot_col*cols;
		workspace_size += n_array*dim;
		workspace_size += n_array;
		workspace_size += 1;
		x_array_size = n_array*dim*sizeof(GT);
		t_array_size = n_array*sizeof(GT);

		cudaMalloc((void **)&all, n_path*workspace_size*sizeof(GT));

		matrix = all +n;
		V = matrix;
		R = matrix+n_matrix;
		sol = R + (dim + 1)*(dim + 1);
		P = sol + dim;
		x_array = P + row_block*matrix_block_pivot_col*cols;
		t_array = x_array+n_array*dim;
		one_minor_t = t_array + n_array;

		coef = all;
		mon = coef + n_coef;
		sum = mon - n_constant;
		f_val = matrix + n_eq*dim;
		x = x_array;
		t = t_array;
	    x_last = x;
	    t_last = t;

	    x_t_idx = 0;

		// GT arrays
		int n_x_t_arrays = n_path*(dim+5);
		int n_eval_arrays = n_path*(n_coef+mon_pos_size+n_eq*(dim+1));
		n_matrix_R = (dim+2)*(dim+1)/2;
		int n_qr_arrays = n_path*(n_eq*(dim+1)+ n_matrix_R + dim);
		int n_predict_arrays = n_path*(dim*(n_predictor+1)+(n_predictor+1));

		n_GT_arrays = n_x_t_arrays+n_eval_arrays+n_qr_arrays+n_predict_arrays;
		size_GT_arrays = n_GT_arrays*sizeof(GT);
		cudaMalloc((void **) &GT_arrays, size_GT_arrays);

		// GT arrays: Eval arrays
		coef_mult = GT_arrays;
		mon_mult = coef_mult + n_coef*n_path;
		sum_mult = mon_mult - n_constant*n_path;
		matrix_mult = mon_mult + mon_pos_size*n_path;
		f_val_mult = matrix_mult + n_eq*dim*n_path;
		// GT arrays: QR arrays
		matrix_mult_horizontal = matrix_mult+n_path*n_eq*(dim+1);
		V_mult = matrix_mult_horizontal;
		R_mult = V_mult+n_matrix*n_path;
		sol_mult = R_mult+n_matrix_R*n_path;
		// GT arrays: Predict arrays
		x_array_mult = sol_mult + dim*n_path;
		t_array_mult = x_array_mult + dim*(n_predictor+1)*n_path;
		// GT arrays: x and t arrays
		x_mult           = t_array_mult + (n_predictor+1)*n_path;
		t_mult           = x_mult + n_path*dim;
		t_last_mult      = t_mult + n_path;
		delta_t_mult     = t_mult + 2*n_path;
		newton_t_mult    = t_mult + 3*n_path;
		one_minor_t_mult = t_mult + 4*n_path;

	    // int arrays
		cudaMalloc((void **) &int_arrays, 7*n_path*sizeof(int));
		x_t_idx_mult   = int_arrays;
		n_point_mult   = int_arrays + n_path;
		path_success   = int_arrays + 2*n_path;
		n_success      = int_arrays + 3*n_path;
		newton_success = int_arrays + 4*n_path;
		path_idx       = int_arrays + 5*n_path;
		end_range      = int_arrays + 6*n_path;

		// double arrays
		cudaMalloc((void **) &double_arrays, 6*n_path*sizeof(double));
		max_delta_x_gpu    = double_arrays;
		r_max_delta_x_gpu  = double_arrays + n_path;
		max_f_val_gpu      = double_arrays + 2*n_path;
		max_f_val_last_gpu = double_arrays + 3*n_path;
		r_max_f_val_gpu    = double_arrays + 4*n_path;
		max_x_gpu          = double_arrays + 5*n_path;

		// host arrays
		path_success_host = (int *)malloc(n_path*sizeof(int));
		newton_success_host = (int *)malloc(n_path*sizeof(int));
		path_idx_host = (int *)malloc(n_path*sizeof(int));

		r_max_f_val_host   = (double *)malloc(n_path*sizeof(double));
		max_f_val_host     = (double *)malloc(n_path*sizeof(double));
		max_delta_x_host   = (double *)malloc(n_path*sizeof(double));
		r_max_delta_x_host = (double *)malloc(n_path*sizeof(double));
		max_x_host         = (double *)malloc(n_path*sizeof(double));

		// eq in one block, developping
		workspace_eq=NULL;

		std::cout << "GPU initialized" << std::endl;
	}

	~GPUWorkspace(){
		std::cout << "Delete GPUWorkspace" << std::endl;
		cudaFree(all);
	}

	void init_workspace_eq(int n_pos_total_eq, int n_path);

	void init_matrix(int dim, int n_eq);

	void init_V_value(CT* V_cpu, int sys_idx=0);

	void init_x_t(int dim, int n_predictor);

	void update_x_t_idx();

	void update_x_t_idx_all(int* x_t_idx_host);

	void update_t_value(CT cpu_t);

	void update_t_value_array(CT* cpu_t, int* x_t_idx_host);

	void update_t_value_mult(CT* cpu_t);

	void update_t_value_inverse(CT cpu_one_minor_t);

	void update_x_value(CT* cpu_sol0);

	void update_x_value_array(CT* cpu_sol0);

	void update_x_value_mult(CT* cpu_sol0);

	void update_x_mult_horizontal(CT* cpu_sol0);

	void update_x_t_value(CT* cpu_sol0, CT cpu_t);

	void update_x_t_value_array(CT* cpu_sol0, CT* cpu_t, int* x_t_idx_host);

	void update_x_t_value_mult(CT* cpu_sol0, CT* cpu_t);

	void update_x_value_mult2(CT* cpu_sol0, int* x_t_idx_host);

	void update_t_value_mult2(CT* cpu_t, int* x_t_idx_host);

	void update_x_t(CT* cpu_sol0, CT cpu_t);

	CT* get_workspace();

	CT* get_matrix();

	CT* get_workspace(int sys_idx);

	CT* get_matrix(int sys_idx);

	CT* get_matrix_r(int sys_idx=0);

	void print_matrix_r();

	CT* get_x();

	CT** get_mult_x_horizontal();

	CT* get_x_array();

	CT* get_t_array();

	CT** get_x_all();

	CT** get_x_last_all();

	void print_x();

	void print_x_mult(int path_idx_one=-1);

	void print_t_mult(int path_idx_one=-1);

	void print_delta_t_mult(int path_idx_one=-1);

	void print_x_last_mult();

	void print_x_array();

	void print_t_array();

	CT* get_workspace_mult();

	CT* get_coef_mult();

	CT* get_mon_mult();

	CT* get_matrix_mult();

	CT* get_x_last();

	CT* get_sol(int sys_idx=0);

	CT* get_sol_mult();

	T1 sol_norm();

	void init_x_t_predict_test();
};

class GPUInst{
public:
	int n_path;

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

	int mon_pos_size;

	int mon_level0_BS;
	int mon_level_BS;

	int mon_global_BS;
	dim3 mon_global_grid;

	int n_mon_block;
	dim3 mon_block_grid;
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

	int* sum_pos_start_align;
	int* sum_pos_align;

	int sum_BS;
	dim3 sum_grid;

	int n_step_GPU;
	int n_point_GPU;
	int n_eval_GPU;
	int n_mgs_GPU;

	int predict_BS;
	dim3 predict_grid;

	int* eq_pos_start;
	int n_mon_total_eq;
	int* mon_pos_start_eq;
	GT* coef_eq;
	int n_pos_total_eq;
	unsigned short* mon_pos_eq;

	CT alpha;

	int n_sum_zero;
	int* sum_zeros;

	GPUInst(const CPUInstHom& cpu_inst, int n_path){
		dim = cpu_inst.dim;
		n_eq = cpu_inst.n_eq;
		this->n_path = n_path;
		init_predict();
		init_coef(cpu_inst.CPU_inst_hom_coef);
		init_mon(cpu_inst.CPU_inst_hom_mon, cpu_inst.CPU_inst_hom_block);
		if(MON_EVAL_METHOD == 1){
			init_sum(cpu_inst.CPU_inst_hom_sum_block);
		}
		else{
			init_sum(cpu_inst.CPU_inst_hom_sum);
		}
		init_workspace(cpu_inst);

		dim_BS = 32;
	    dim_grid = get_grid(dim,dim_BS,n_path);
	    n_step_GPU = 0;
	    n_point_GPU = 0;
	    n_eval_GPU = 0;
	    n_mgs_GPU = 0;

	    init_eq(cpu_inst.CPU_inst_hom_eq);
	}

	~GPUInst(){
		cudaFree(coef);
		cudaFree(mon_pos_start);
		cudaFree(mon_pos);
		cudaFree(sum_pos_start);
		cudaFree(sum_pos);
	}

	void init_predict();

	void init_coef(const CPUInstHomCoef& cpu_inst_coef);

	void init_mon(const CPUInstHomMon& cpu_inst_mon, const CPUInstHomMonBlock& cpu_inst_mon_block);

	void init_sum(const CPUInstHomSumBlock& cpu_inst_sum);

	void init_sum(const CPUInstHomSum& cpu_inst_sum);

	void init_workspace(const CPUInstHom& cpu_inst);

	void init_eq(const CPUInstHomEq& cpu_inst_mon_eq);
};

#endif /* GPU_DATA_H_ */

