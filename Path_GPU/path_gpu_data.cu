#ifndef __GPU_DATA_CU_
#define __GPU_DATA_CU_

#include "path_gpu_data.h"

int get_NB(int n_job, int BS, int n_thread_per_job){
	return (n_job*n_thread_per_job - 1)/BS + 1;
}

dim3 get_grid(int NB, int n_sys){
	int NBy = 1;
	int NBx = NB;
	while(NBx > 65535){
		NBy++;
		NBx = (NB-1)/NBy + 1;
	}
	return dim3 (NBx,NBy,n_sys);
}

dim3 get_grid(int n_job, int BS, int n_sys, int n_thread_per_job){
	int NB = get_NB(n_job, BS, n_thread_per_job);
	return get_grid(NB, n_sys);
}

void GPUWorkspace::init_matrix(int dim, int n_eq){
	V_size = size_matrix;
	V = matrix;
	R_size = (dim + 1)*(dim + 1)*sizeof(GT);
	cudaMalloc((void **)&R, R_size);
	sol_size = dim*sizeof(GT);
	std::cout << "dim = " << dim << " " << sol_size << std::endl;
	cudaMalloc((void **)&sol, sol_size);

	int rows = n_eq;
	int cols = dim+1;
	int row_block = (rows-1)/matrix_block_row+1;
	cudaMalloc((void**)&P, row_block*matrix_block_pivot_col*cols*sizeof(GT));
}

void GPUWorkspace::init_V_value(CT* V_cpu){
	GT* V_host = (GT*)malloc(V_size);
	//std::cout << "----------------n_eq = " << n_eq << " dim = " << dim << std::endl;
	for(int i=0; i<n_eq*(dim+1); i++){
    	comp1_qd2gqd(&V_cpu[i],&V_host[i]);
	}
	cudaMemcpy(V, V_host, V_size, cudaMemcpyHostToDevice);
}

void GPUWorkspace::init_x_t(int dim, int n_predictor){
	x_t_idx = 0;
	this->n_predictor = n_predictor;
	this->dim = dim;
	n_array = n_predictor + 1;

	x_array_size = n_array*dim*sizeof(GT);
	cudaMalloc((void **)&x_array, x_array_size);

	t_array_size = n_array*sizeof(GT);
	cudaMalloc((void **)&t_array, t_array_size);

	x = x_array;
	t = t_array;

	cudaMalloc((void **)&one_minor_t, sizeof(GT));
}

void GPUWorkspace::update_x_t_idx(){
    x_last = x;
    t_last = t;
    x_t_idx = (x_t_idx+1)%n_array;
    x = x_array + dim*x_t_idx;
    t = t_array + x_t_idx;
}

void GPUWorkspace::update_t_value(CT cpu_t){
	CT cpu_one_minor_t(1.0-cpu_t.real, -cpu_t.imag);
	cpu_one_minor_t *= alpha;

	GT* host_t = (GT*)malloc(sizeof(GT));
	comp1_qd2gqd(&cpu_t, host_t);
	cudaMemcpy(t, host_t, sizeof(GT), cudaMemcpyHostToDevice);
	free(host_t);

    GT* host_one_minor_t = (GT*)malloc(sizeof(GT));
    comp1_qd2gqd(&cpu_one_minor_t, host_one_minor_t);
	cudaMemcpy(one_minor_t, host_one_minor_t, sizeof(GT), cudaMemcpyHostToDevice);
	free(host_one_minor_t);
}


void GPUWorkspace::update_t_value_inverse(CT cpu_one_minor_t){
	CT cpu_t(1.0-cpu_one_minor_t.real, -cpu_one_minor_t.imag);
	cpu_one_minor_t *= alpha;

	GT* host_t = (GT*)malloc(sizeof(GT));
	comp1_qd2gqd(&cpu_t, host_t);
	cudaMemcpy(t, host_t, sizeof(GT), cudaMemcpyHostToDevice);
	free(host_t);

    GT* host_one_minor_t = (GT*)malloc(sizeof(GT));
    comp1_qd2gqd(&cpu_one_minor_t, host_one_minor_t);
	cudaMemcpy(one_minor_t, host_one_minor_t, sizeof(GT), cudaMemcpyHostToDevice);
	free(host_one_minor_t);
}

void GPUWorkspace::update_x_value(CT* cpu_sol0){
	GT* host_sol0 = (GT*)malloc(dim*sizeof(GT));
    for(int i=0; i<dim; i++){
    	comp1_qd2gqd(&cpu_sol0[i],&host_sol0[i]);
    }
	size_t sol0_size = dim*sizeof(GT);
	cudaMemcpy(x_array, host_sol0, sol0_size, cudaMemcpyHostToDevice);
	free(host_sol0);
}

void GPUWorkspace::update_x_t_value(CT* cpu_sol0, CT cpu_t){
	update_x_value(cpu_sol0);
	update_t_value(cpu_t);
}

void GPUWorkspace::update_x_t(CT* cpu_sol0, CT cpu_t){
	update_x_t_value(cpu_sol0, cpu_t);
	update_x_t_idx();
}

CT* GPUWorkspace::get_workspace(){
    GT* host_workspace = (GT*)malloc(size_all);
	cudaMemcpy(host_workspace, all, size_all, cudaMemcpyDeviceToHost);
	//std::cout << "****n = " << n << std::endl;
	CT* gpu_workspace = (CT*)malloc(size_all);
    for(int i=0; i<n; i++) comp1_gqd2qd(&host_workspace[i], &gpu_workspace[i]);

    free(host_workspace);
    return gpu_workspace;
}

CT* GPUWorkspace::get_matrix(){
    GT* host_matrix = (GT*)malloc(size_matrix);
	cudaMemcpy(host_matrix, matrix, size_matrix, cudaMemcpyDeviceToHost);

	CT* gpu_matrix = (CT*)malloc(size_matrix);
    for(int i=0; i<n_matrix; i++) comp1_gqd2qd(&host_matrix[i], &gpu_matrix[i]);

    free(host_matrix);
    return gpu_matrix;
}

CT* GPUWorkspace::get_matrix_r(){
    GT* host_matrix_r = (GT*)malloc(R_size);
	cudaMemcpy(host_matrix_r, R, R_size, cudaMemcpyDeviceToHost);

	CT* gpu_matrix_r = (CT*)malloc(R_size);
    for(int i=0; i<(dim + 1)*(dim + 1); i++)
    	comp1_gqd2qd(&host_matrix_r[i], &gpu_matrix_r[i]);

    free(host_matrix_r);
    return gpu_matrix_r;
}

void GPUWorkspace::print_matrix_r(){
	CT* R = get_matrix_r();
	int tmp_idx_r = 0;
	for(int i=0; i<dim+1; i++){
		for(int j=0; j<dim+1-i; j++){
			if(i!=0 or j!= dim){
				std::cout << dim-i << " " << j << " " << R[tmp_idx_r];
			}
			tmp_idx_r++;
		}
	}
}

CT* GPUWorkspace::get_x(){
	size_t x_size = dim*sizeof(GT);
    GT* x_host = (GT*)malloc(x_size);
	cudaMemcpy(x_host, x, x_size, cudaMemcpyDeviceToHost);

	CT* x_cpu = (CT*)malloc(x_size);
    for(int i=0; i<dim; i++) comp1_gqd2qd(&x_host[i], &x_cpu[i]);

    free(x_host);
    return x_cpu;
}

CT* GPUWorkspace::get_x_array(){
    GT* x_array_host = (GT*)malloc(x_array_size);
	cudaMemcpy(x_array_host, x_array, x_array_size, cudaMemcpyDeviceToHost);

	CT* x_array_ct = (CT*)malloc(x_array_size);
    for(int i=0; i<n_array*dim; i++) comp1_gqd2qd(&x_array_host[i], &x_array_ct[i]);

    free(x_array_host);
    return x_array_ct;
}


void GPUWorkspace::print_x(){
	CT* x_ct = get_x();
	for(int i=0; i<dim; i++){
		std::cout << i << " "  << x_ct[i];
	}
	free(x_ct);
}

void GPUWorkspace::print_x_array(){
	CT* x_array_ct = get_x_array();
	for(int i=0; i<dim; i++){
		for(int j=0; j<n_array; j++){
			std::cout << i << " " << j << " " << x_array_ct[j*dim+i];
		}
		std::cout << std::endl;
	}
	free(x_array_ct);
}

CT* GPUWorkspace::get_x_last(){
	size_t x_size = dim*sizeof(GT);
    GT* x_host = (GT*)malloc(x_size);
	cudaMemcpy(x_host, x_last, x_size, cudaMemcpyDeviceToHost);

	CT* x_cpu = (CT*)malloc(x_size);
    for(int i=0; i<dim; i++) comp1_gqd2qd(&x_host[i], &x_cpu[i]);

    free(x_host);
    return x_cpu;
}

CT* GPUWorkspace::get_sol(){
	size_t sol_size = dim*sizeof(GT);
    GT* sol_host = (GT*)malloc(sol_size);
	cudaMemcpy(sol_host, sol, sol_size, cudaMemcpyDeviceToHost);

	CT* sol_cpu = (CT*)malloc(sol_size);
    for(int i=0; i<dim; i++){
    	comp1_gqd2qd(&sol_host[i], &sol_cpu[i]);
    	//std::cout << i << " " << sol_cpu[i] << std::endl;
    }

    free(sol_host);
    return sol_cpu;
}

T1 GPUWorkspace::sol_norm(){
    GT* sol_host = (GT*)malloc(sol_size);
	cudaMemcpy(sol_host, sol, sol_size, cudaMemcpyDeviceToHost);

	CT tmp_sol;
	comp1_gqd2qd(&sol_host[0], &tmp_sol);
	T1 max_delta = tmp_sol.real*tmp_sol.real+tmp_sol.imag*tmp_sol.imag;
	for(int k=1; k<dim; k++){
		comp1_gqd2qd(&sol_host[0], &tmp_sol);
		T1 tmp_delta = tmp_sol.real*tmp_sol.real+tmp_sol.imag*tmp_sol.imag;
		if(tmp_delta>max_delta){
			max_delta = tmp_delta;
		}
	}
	return max_delta;
}


void GPUWorkspace::init_x_t_predict_test(){
	std::cout << "--------- Initializa x and t value for Testing only --------- "\
			  <<std::endl;
	CT* x_array_cpu = (CT*)malloc(n_predictor*dim*sizeof(CT));
	CT* t_array_cpu = (CT*)malloc(n_predictor*sizeof(CT));
	for(int i=0; i<n_predictor; i++){
		for(int j=0; j<dim; j++){
			x_array_cpu[i*dim + j] = CT(((i+1)*(i+1)*(i+1)+5),0);
			std::cout << i << " " << j << " " << x_array_cpu[i*dim + j];
		}
		t_array_cpu[i] = CT(i+1,0);
		std::cout << i << " " << t_array_cpu[i];
	}

	GT* x_array_host = (GT*)malloc(n_predictor*dim*sizeof(GT));
    for(int i=0; i<dim*n_predictor; i++){
    	comp1_qd2gqd(&x_array_cpu[i],&x_array_host[i]);
    }

	GT* t_array_host = (GT*)malloc(n_predictor*sizeof(GT));
    for(int i=0; i<n_predictor; i++){
    	comp1_qd2gqd(&t_array_cpu[i],&t_array_host[i]);
    }

	cudaMemcpy(x_array+dim, x_array_host, n_predictor*dim*sizeof(GT), cudaMemcpyHostToDevice);
	cudaMemcpy(t_array+1, t_array_host, n_predictor*sizeof(GT), cudaMemcpyHostToDevice);

    free(x_array_cpu);
    free(t_array_cpu);
    free(x_array_host);
    free(t_array_host);

}

void GPUInst::init_coef(const CPUInstHomCoef& cpu_inst_coef){
	n_coef = cpu_inst_coef.n_coef;
	size_t coef_size = n_coef*2*sizeof(GT);

    GT* host_coef = (GT *)malloc(coef_size);
    CT* tmp_cpu_coef = cpu_inst_coef.coef_orig;
    for(int i=0; i<n_coef*2; i++){
    	comp1_qd2gqd(&tmp_cpu_coef[i],&host_coef[i]);
    }

	cudaMalloc((void **)&coef, coef_size);
	cudaMemcpy(coef, host_coef, coef_size, cudaMemcpyHostToDevice);
	free(host_coef);

	coef_BS = 64;
    coef_grid = get_grid(n_coef, coef_BS, n_sys);

    alpha = cpu_inst_coef.alpha;
}

void GPUInst::init_mon(const CPUInstHomMon& cpu_inst_mon, const CPUInstHomMonBlock& cpu_inst_mon_block){
	n_mon_block = cpu_inst_mon_block.n_mon;
	BS_mon_block = cpu_inst_mon_block.BS;
	NB_mon_block = cpu_inst_mon_block.NB;
	//mon_pos_block_size = cpu_inst_mon_block.mon_pos_block_size;

	size_t mon_pos_start_block_size = NB_mon_block*sizeof(int);
	cudaMalloc((void **)&mon_pos_start_block, mon_pos_start_block_size);
	cudaMemcpy(mon_pos_start_block, cpu_inst_mon_block.mon_pos_start_block, mon_pos_start_block_size, cudaMemcpyHostToDevice);

	size_t mon_pos_block_size = cpu_inst_mon_block.mon_pos_block_size*sizeof(unsigned short);
	cudaMalloc((void **)&mon_pos_block, mon_pos_block_size);
	cudaMemcpy(mon_pos_block, cpu_inst_mon_block.mon_pos_block, mon_pos_block_size, cudaMemcpyHostToDevice);

	level = cpu_inst_mon.level;
	n_mon_level = cpu_inst_mon.n_mon_level;
	n_mon = cpu_inst_mon.n_mon;

	/*std::cout << "n_mon = " << n_mon << " level = " << level << std::endl;
	for(int i=0; i<level; i++){
		std::cout << "level " << i << " " << n_mon_level[i] << std::endl;
	}*/

	size_t mon_pos_start_size = n_mon*sizeof(int);
	cudaMalloc((void **)&mon_pos_start, mon_pos_start_size);
	cudaMemcpy(mon_pos_start, cpu_inst_mon.mon_pos_start, mon_pos_start_size, cudaMemcpyHostToDevice);

	size_t mon_pos_size = cpu_inst_mon.mon_pos_size*sizeof(unsigned short);
	cudaMalloc((void **)&mon_pos, mon_pos_size);
	cudaMemcpy(mon_pos, cpu_inst_mon.mon_pos, mon_pos_size, cudaMemcpyHostToDevice);

	mon_level_grid = new dim3[level];
	// rest monomial after this level use global method
	n_mon_level_rest = new int[level];
	mon_level_grid_rest = new dim3[level];

	mon_level0_BS = 32;
	mon_global_BS = 64;
    mon_level_grid[0] = get_grid(n_mon_level[0],mon_level0_BS,n_sys);
    n_mon_level_rest[0] = n_mon -n_mon_level[0];
    mon_level_grid_rest[0] = get_grid(n_mon_level_rest[0],mon_global_BS,n_sys);

    mon_level_grid[1] = get_grid(n_mon_level[1],mon_global_BS,n_sys);
    n_mon_level_rest[1] = n_mon_level_rest[0] -n_mon_level[1];
    mon_level_grid_rest[1] = get_grid(n_mon_level_rest[1],mon_global_BS,n_sys);
	// for level 1
	// for dd cyclic 96
	// 64 is the best
	// 32 128 has similar result
	// 16 is bad

    mon_level_BS = shmemsize/2;
	int n_thread_per_job = 2;
	for(int i=2; i<level; i++){
	    mon_level_grid[i] = get_grid(n_mon_level[i],mon_level_BS,n_sys, n_thread_per_job);
	    n_mon_level_rest[i] = n_mon_level_rest[i-1] -n_mon_level[i];
	    mon_level_grid_rest[i] = get_grid(n_mon_level_rest[i],mon_global_BS,n_sys, n_thread_per_job);
	    n_thread_per_job *= 2;
	}

	n_mon_global = n_mon - n_mon_level[0];
    mon_global_grid = get_grid(n_mon_global,mon_global_BS,n_sys);
}

void GPUInst::init_sum(const CPUInstHomSumBlock& cpu_inst_sum){
	n_sum = cpu_inst_sum.n_sum;
	n_sum_levels = cpu_inst_sum.n_sum_levels;
	n_sum_level= cpu_inst_sum.n_sum_level;
	n_sum_level_rest = cpu_inst_sum.n_sum_level_rest;

	size_t sum_pos_start_size = n_sum*sizeof(int);
	cudaMalloc((void **)&sum_pos_start, sum_pos_start_size);
	cudaMemcpy(sum_pos_start, cpu_inst_sum.sum_pos_start, sum_pos_start_size, cudaMemcpyHostToDevice);

	size_t sum_pos_size = cpu_inst_sum.sum_pos_size*sizeof(int);
	cudaMalloc((void **)&sum_pos, sum_pos_size);
	cudaMemcpy(sum_pos, cpu_inst_sum.sum_pos, sum_pos_size, cudaMemcpyHostToDevice);

	sum_BS = 32;
    sum_grid = get_grid(n_sum,sum_BS,n_sys);

	sum_level_grid = new dim3[n_sum_levels];
	sum_level_grid_rest = new dim3[n_sum_levels];

	int n_thread_per_job = 1;

	sum_level_grid[0] = get_grid(n_sum_level[0], sum_BS, n_sys, n_thread_per_job);
	sum_level_grid_rest[0] = get_grid(n_sum, sum_BS, n_sys, n_thread_per_job);
	n_thread_per_job *= 2;

	for(int i=1; i<n_sum_levels; i++){
		sum_level_grid[i] = get_grid(n_sum_level[i], sum_BS, n_sys, n_thread_per_job);
		sum_level_grid_rest[i] = get_grid(n_sum_level_rest[i-1], sum_BS, n_sys, n_thread_per_job);
		n_thread_per_job *= 2;
	}
}

void GPUInst::init_sum(const CPUInstHomSum& cpu_inst_sum){
	n_sum = cpu_inst_sum.n_sum;
	n_sum_levels = cpu_inst_sum.n_sum_levels;
	n_sum_level= cpu_inst_sum.n_sum_level;
	n_sum_level_rest = cpu_inst_sum.n_sum_level_rest;

	size_t sum_pos_start_size = n_sum*sizeof(int);
	cudaMalloc((void **)&sum_pos_start, sum_pos_start_size);
	cudaMemcpy(sum_pos_start, cpu_inst_sum.sum_pos_start, sum_pos_start_size, cudaMemcpyHostToDevice);

	size_t sum_pos_size = cpu_inst_sum.sum_pos_size*sizeof(int);
	cudaMalloc((void **)&sum_pos, sum_pos_size);
	cudaMemcpy(sum_pos, cpu_inst_sum.sum_pos, sum_pos_size, cudaMemcpyHostToDevice);

	sum_BS = 32;
    sum_grid = get_grid(n_sum,sum_BS,n_sys);

	sum_level_grid = new dim3[n_sum_levels];
	sum_level_grid_rest = new dim3[n_sum_levels];

	int n_thread_per_job = 1;

	sum_level_grid[0] = get_grid(n_sum_level[0], sum_BS, n_sys, n_thread_per_job);
	sum_level_grid_rest[0] = get_grid(n_sum, sum_BS, n_sys, n_thread_per_job);
	n_thread_per_job *= 2;

	for(int i=1; i<n_sum_levels; i++){
		sum_level_grid[i] = get_grid(n_sum_level[i], sum_BS, n_sys, n_thread_per_job);
		sum_level_grid_rest[i] = get_grid(n_sum_level_rest[i-1], sum_BS, n_sys, n_thread_per_job);
		n_thread_per_job *= 2;
	}
}

void GPUInst::init_workspace(const CPUInstHom& cpu_inst){
	//std::cout << "n_workspace1 = " << n_workspace1 << std::endl;
	n_workspace = n_coef + cpu_inst.CPU_inst_hom_block.mon_pos_block_size + n_mon_level[0]*2;
	//n_workspace = n_coef + cpu_inst.CPU_inst_hom_mon.mon_pos_size;
	n_constant = cpu_inst.n_constant;
};

#endif /*__GPU_DATA_CU__*/
