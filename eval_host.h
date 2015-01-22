/*
 * CPU_instruction_eval.h
 *
 *  Created on: Nov 25, 2014
 *      Author: yxc
 */

#ifndef CPU_INSTRUCTION_EVAL_H_
#define CPU_INSTRUCTION_EVAL_H_

#include "complex.h"
#include "varset.h"
#include "utilities.h"
#include "poly.h"
#include "workspace_host.h"
#include <sys/time.h>
#include <unistd.h>

class CPUInstHomCoef{
  public:
	int n_coef; // *2 = coef_orig size
	CT* coef_orig;
	CT alpha;

	CPUInstHomCoef(){
		n_coef = 0;
		coef_orig = NULL;
		alpha = CT(0.0,0);
	}

	CPUInstHomCoef(MonSet* hom_monset, int total_n_mon, int n_monset, int n_constant, CT alpha){
		init(hom_monset, total_n_mon, n_monset, n_constant, alpha);
	}

	void init(MonSet* hom_monset, int total_n_mon, int n_monset, int n_constant, CT alpha){
		this->alpha = alpha;
		n_coef = total_n_mon;
		//std::cout << "n_coef = " << n_coef << std::endl;
		coef_orig = new CT[n_coef*2];
		CT* tmp_coef_orig = coef_orig;

		int constant_exist = 0;
		if(n_constant > 0){
			constant_exist = 1;
		}

		for(int i=constant_exist; i<n_monset; i++){
			hom_monset[i].write_coef(tmp_coef_orig);
		}

		if(n_constant > 0){
			hom_monset[0].write_coef(tmp_coef_orig);
		}
	}

	~CPUInstHomCoef(){
		std::cout << "Delete CPUInstHomCoef" << std::endl;
		delete[] coef_orig;
	}

	void print(){
		//Print coefficient
		for(int i=0; i<n_coef; i++){
		std::cout << i << std::endl
		          << coef_orig[2*i]
		          << coef_orig[2*i+1]<< std::endl;
		}
	}

	void eval(const CT t, CT* coef, int reverse=0){
		CT one_minor_t(1.0- t.real, -t.imag);

		//std::cout << "one_minor_t = " << one_minor_t << std::endl;
		/*CT a(-9.85602019157774E-01  -1.69081814019484E-01);
		int k = 2;
		CT t_power_k = t;
		CT one_minor_t_power_k = a*one_minor_t;
		for(int i=1; i<k; i++){
			t_power_k *= t;
			one_minor_t_power_k *= one_minor_t;
		}*/

		CT t0, t1;
		if(reverse == 0){
			t0 = one_minor_t*alpha;
			t1 = t;
		}
		else{
			t0 = t*alpha;
			t1 = one_minor_t;
		}

		for(int i=0; i<n_coef; i++){
			//coef[i] = coef_orig[2*i]*t_power_k + coef_orig[2*i+1]*one_minor_t_power_k;
			//alpha*(1-t)*P + t*Q
			coef[i] = coef_orig[2*i+1]*t0 + coef_orig[2*i]*t1 ;
		}
	}

	void update_alpha();
};

class CPUInstHomMon{
  public:
	int level; // size of n_mon_level
	int* n_mon_level;
	int n_mon; // size of pos_start, sum of n_mon_level
	int* mon_pos_start;
	int mon_pos_size;
	unsigned short* mon_pos;

	CPUInstHomMon(){
		level = 0;
		n_mon_level = NULL;
		n_mon = 0;
		mon_pos_start = NULL;
		mon_pos_size = 0;
		mon_pos = NULL;
	}

	CPUInstHomMon(MonSet* hom_monset, int n_monset, int total_n_mon, int n_constant){
		init(hom_monset, n_monset, total_n_mon, n_constant);
	}

	void init(MonSet* hom_monset, int n_monset, int total_n_mon, int n_constant)
	{
		int max_n_var = hom_monset[n_monset-1].get_n();
	    level = 1;
	    for(int i=1; i<max_n_var; i<<=1){
	        level++;
	    }

	    n_mon_level = new int[level];
	    for(int i=0; i<level; i++){
	    	n_mon_level[i] = 0;
	    }

		mon_pos_size = 0;

		// XXX Only work for single monomial
		n_mon = total_n_mon - n_constant;

		int constant_exist;
		if(n_constant == 0  ){
			constant_exist = 0;
		}
		else{
			constant_exist = 1;
		}

		//Write monomial start position
		int tmp_level = 0;
		int tmp_level_size = 1;
		mon_pos_start = new int[n_mon];

		// XXX Only work for single monomial
		int mon_idx = 0;
		for(int i=constant_exist; i<n_monset; i++){
			int tmp_n = hom_monset[i].get_n();
			int tmp_n_mon = hom_monset[i].get_n_mon();
			for(int j=0; j<tmp_n_mon; j++){
				mon_pos_start[mon_idx++] = mon_pos_size;
				mon_pos_size += tmp_n+1;
			}
			while(tmp_level_size < tmp_n){
				tmp_level_size *= 2;
				tmp_level++;
			}
			n_mon_level[tmp_level]+=tmp_n_mon;
		}

		//Write position instruction
		mon_pos = new unsigned short[mon_pos_size];

		unsigned short* tmp_pos = mon_pos;
		for(int i=constant_exist; i<n_monset; i++){
			// Number of variable each term
			int tmp_n_mon = hom_monset[i].get_n_mon();
			for(int j=0; j<tmp_n_mon; j++){
				*tmp_pos++ = hom_monset[i].get_n();
				// Write position
				hom_monset[i].write_pos(tmp_pos);
			}
		}
	}

	~CPUInstHomMon(){
		std::cout << "Delete CPUInstHomMon" << std::endl;
		delete[] n_mon_level;
		delete[] mon_pos_start;
		delete[] mon_pos;
	}

	void print(){
		std::cout << "level = " << level << std::endl;
		for(int i=0; i<level; i++){
			std::cout << i << " " << n_mon_level[i] << std::endl;
		}

		std::cout << "n_mon = " << n_mon << std::endl;
		//Print monomial with position
		for(int i=0; i<n_mon; i++){
			int tmp_n = mon_pos[mon_pos_start[i]];
			std::cout << i << " n = " << tmp_n << ": ";
			for(int j=0; j<tmp_n; j++){
				std::cout << mon_pos[mon_pos_start[i]+j+1] << ", ";
			}
			std::cout << " mon_pos_start = " << mon_pos_start[i] << std::endl;
		}
	}

	void eval(const CT* sol, CT* mon, CT* coef){
		int* tmp_mon_pos_start = mon_pos_start;
		CT* tmp_coef = coef;

		for(int j=0; j<n_mon_level[0]; j++){
			//std::cout << " j = " << j << " tmp_mon_pos_start[j] = " << tmp_mon_pos_start[j] << std::endl;
			int tmp_idx = tmp_mon_pos_start[j];
			cpu_speel0(sol, mon_pos+tmp_idx, mon+tmp_idx, tmp_coef[j]);
		}
		tmp_mon_pos_start += n_mon_level[0];
		tmp_coef += n_mon_level[0];

		for(int i=1; i<level; i++){
			//std::cout << "level = " << i << " n_mon_level = " << n_mon_level[i] << std::endl;
			for(int j=0; j<n_mon_level[i]; j++){
				//std::cout << " i = " << i << " j = " << j << " tmp_mon_pos_start[j] = "
				//		  << tmp_mon_pos_start[j] << std::endl;
				int tmp_idx = tmp_mon_pos_start[j];
				cpu_speel(sol, mon_pos+tmp_idx, mon+tmp_idx, tmp_coef[j]);
			}
			tmp_mon_pos_start += n_mon_level[i];
			tmp_coef += n_mon_level[i];
		}
	}
};

class CPUInstHomSum{
  public:
	int n_sum; // size of sum_start
	/*int n_sum0;
	int n_sum2;
	int n_sum4;
	int n_sum8;
	int n_sum_n;*/
	int n_sum_levels;
	int* n_sum_level;
	int* n_sum_level_rest;
	int* sum_pos_start;
	int sum_pos_size;
	int* sum_pos;

	CPUInstHomSum(){
		n_sum = 0;
		/*n_sum0 = 0;
		n_sum2 = 0;
		n_sum4 = 0;
		n_sum8 = 0;
		n_sum_n = 0;*/
		n_sum_levels = 0;
		n_sum_level = NULL;
		n_sum_level_rest = NULL;
		sum_pos_start = NULL;
		sum_pos_size = 0;
		sum_pos = NULL;
	}

	CPUInstHomSum(MonSet* hom_monset, int n_monset, const int* mon_pos_start,
			             int dim, int n_eq, int n_constant){
		init(hom_monset, n_monset, mon_pos_start, dim, n_eq, n_constant);
	}

	void init(MonSet* hom_monset, int n_monset, const int* mon_pos_start,
			             int dim, int n_eq, int n_constant)
	{
		std::cout << "dim = " << dim << " n_eq = " << n_eq << std::endl;

		// Step 1: count number of terms to sum in Jacobian matrix
		int* n_sums_loc = new int[n_eq*(dim+1)];
		for(int i=0; i<n_eq*(dim+1); i++){
			n_sums_loc[i] = 0;
		}
		int** n_sums = new int*[n_eq];
		int* n_sums_tmp = n_sums_loc;
		for(int i=0; i<n_eq; i++){
			n_sums[i] = n_sums_tmp;
			n_sums_tmp += dim+1;
		}

		MonSet* tmp_hom_monset = hom_monset;
		for(int i=0; i<n_monset; i++){
			for(int j=0; j<tmp_hom_monset->get_n_mon(); j++){
				int tmp_eq_idx = tmp_hom_monset->get_eq_idx(j);
				n_sums[tmp_eq_idx][dim] += 1;
				for(int k=0; k<tmp_hom_monset->get_n(); k++){
					n_sums[tmp_eq_idx][tmp_hom_monset->get_pos(k)] += 1;
				}
			}
			tmp_hom_monset++;
		}

		// Step 2: Count number of sums of certain number of terms
		//               total number of terms to sum
		//               max number of terms to sum
		int max_n_sums = 0;
		for(int i=0; i<n_eq; i++){
			for(int j=0; j<dim+1; j++){
				if(n_sums[i][j] > max_n_sums){
					max_n_sums = n_sums[i][j];
				}
				//std::cout << n_sums[i][j] << " ";
			}
			//std::cout << std::endl;
		}
		//std::cout << "max_n_sums = " << max_n_sums << std::endl;

		int* n_sums_count = new int[max_n_sums+1];
		for(int i=0; i<max_n_sums+1; i++){
			n_sums_count[i] = 0;
		}
		for(int i=0; i<n_eq; i++){
			for(int j=0; j<dim+1; j++){
				n_sums_count[n_sums[i][j]]++;
			}
		}

		/*n_sum0 = n_sums_count[1];
		n_sum2 = n_sums_count[2] + n_sums_count[3];
		n_sum4 = n_sums_count[4] + n_sums_count[5] + n_sums_count[6] + n_sums_count[7];
		n_sum8 = n_sums_count[8] + n_sums_count[9] + n_sums_count[10] + n_sums_count[11] \
		         +n_sums_count[12] + n_sums_count[13] + n_sums_count[14] + n_sums_count[15];  */                                                                 ;
		n_sum = 0;
		for(int i=0; i<max_n_sums+1; i++){
			n_sum += n_sums_count[i];
			//std::cout << i << " " << n_sums_count[i] << std::endl;
		}
		//n_sum_n = n_sum - n_sum2 - n_sum0 - n_sum4 - n_sum8;

		n_sum_levels = log2ceil(max_n_sums);
		n_sum_level = new int[n_sum_levels];
		n_sum_level_rest = new int[n_sum_levels];

		for(int i=0; i<n_sum_levels; i++){
			n_sum_level[i] = 0;
			n_sum_level_rest[i] = 0;
		}

		n_sum_level[0] = n_sums_count[1];

		int tmp_level_size = 4;
		int tmp_level = 1;

		for(int i=2; i<max_n_sums+1; i++){
			if(tmp_level_size < i){
				tmp_level_size *= 2;
				tmp_level++;
			}
			n_sum_level[tmp_level] += n_sums_count[i];
		}

		std::cout << "n_sum = " << n_sum << std::endl;
		n_sum_level_rest[0] = n_sum - n_sum_level[0];
		std::cout << 0 << " " << n_sum_level[0] << " " << n_sum_level_rest[0] << std::endl;
		for(int i=1; i<n_sum_levels; i++){
			n_sum_level_rest[i] = n_sum_level_rest[i-1] - n_sum_level[i];
			std::cout << i << " " << n_sum_level[i] << " " << n_sum_level_rest[i] << std::endl;
		}

		// sum start
		sum_pos_start = new int[n_sum];
		int tmp_idx = 0;
		int last_length = 0;
		for(int i=1; i<max_n_sums+1; i++){
			for(int j=0; j<n_sums_count[i]; j++){
				if(tmp_idx == 0){
					sum_pos_start[0] = 0;
				}
				else{
					sum_pos_start[tmp_idx] = sum_pos_start[tmp_idx-1] + last_length;
				}
				tmp_idx++;
				last_length = i+2;
			}
		}

		/*for(int i=0; i<n_sum; i++){
			std::cout << i << " " << sum_pos_start[i] << std::endl;
		}*/

		// Start pos of sums
		int* n_sums_start = new int[max_n_sums+1];
		n_sums_start[0] = 0;
		n_sums_start[1] = 0;
		for(int i=2; i<max_n_sums+1; i++){
			n_sums_start[i] = n_sums_start[i-1] + n_sums_count[i-1]*(1+i);
		}

		sum_pos_size = n_sums_start[max_n_sums] + n_sums_count[max_n_sums]*(2+max_n_sums);

		std::cout << "sum_pos_size = " << sum_pos_size << std::endl;

		int* sum_pos_start_loc = new int[n_eq*(dim+1)];
		for(int i=0; i<n_eq*(dim+1); i++){
			sum_pos_start_loc[i] = 0;
		}
		int** sum_pos_start_matrix = new int*[n_eq];
		int* sum_pos_start_matrix_tmp = sum_pos_start_loc;
		for(int i=0; i<n_eq; i++){
			sum_pos_start_matrix[i] = sum_pos_start_matrix_tmp;
			sum_pos_start_matrix_tmp += dim+1;
		}

		sum_pos = new int[sum_pos_size];
		for(int i=0; i<sum_pos_size; i++){
			sum_pos[i] = 0;
		}

		for(int i=0; i<n_eq; i++){
			for(int j=0; j<dim+1; j++){
				int tmp_n = n_sums[i][j];
				int tmp_start = n_sums_start[tmp_n];
				//std::cout << i << " " << j << " " << "tmp_start = " << tmp_start << std::endl;
				sum_pos[tmp_start] = tmp_n;
				sum_pos_start_matrix[i][j] = tmp_start+1;
				sum_pos[tmp_start+tmp_n+1] = j*n_eq + i;
				//sum_pos[tmp_start+tmp_n+1] = i*(dim+1) + j;
				n_sums_start[tmp_n] += tmp_n+2;
			}
		}

		/*for(int i=0; i<n_eq; i++){
			for(int j=0; j<dim+1; j++){
				std::cout << sum_pos_start[i][j] << " ";
			}
			std::cout << std::endl;
		}*/

		tmp_hom_monset = hom_monset;
		for(int i=0; i<tmp_hom_monset->get_n_mon(); i++){
			int tmp_eq_idx = tmp_hom_monset->get_eq_idx(i);
			sum_pos[sum_pos_start_matrix[tmp_eq_idx][dim]] = i;
			sum_pos_start_matrix[tmp_eq_idx][dim]++;
		}

		// XXX Only work for single monomial, repeat monomial doesn't work
		tmp_hom_monset = hom_monset+1;
		int mon_idx = 0;
		for(int i=1; i<n_monset; i++){
			//std::cout << *tmp_hom_monset;
			// XXX Only work for single monomial, repeat monomial doesn't work
			for(int j=0; j<tmp_hom_monset->get_n_mon(); j++){
				int tmp_pos = mon_pos_start[mon_idx++]+n_constant;
				int tmp_eq_idx = tmp_hom_monset->get_eq_idx(j);
				// Value
				sum_pos[sum_pos_start_matrix[tmp_eq_idx][dim]] = tmp_pos;
				tmp_pos++;
				sum_pos_start_matrix[tmp_eq_idx][dim]++;
				n_sums[tmp_eq_idx][dim] += 1;
				// Derivative
				for(int k=0; k<tmp_hom_monset->get_n(); k++){
					sum_pos[sum_pos_start_matrix[tmp_eq_idx][tmp_hom_monset->get_pos(k)]] = tmp_pos;
					tmp_pos++;
					sum_pos_start_matrix[tmp_eq_idx][tmp_hom_monset->get_pos(k)]++;
				}
			}
			tmp_hom_monset++;
		}

		delete[] n_sums;
		delete[] n_sums_loc;
		delete[] n_sums_count;
		delete[] n_sums_start;
		delete[] sum_pos_start_loc;
		delete[] sum_pos_start_matrix;
	}

	~CPUInstHomSum(){
		std::cout << "Delete CPUInstHomSum" << std::endl;
		delete[] sum_pos_start;
		delete[] sum_pos;
	}

	void print(){
		std::cout << "n_sum = " << n_sum << std::endl;
		std::cout << "sum_pos_size = " << sum_pos_size << std::endl;
		for(int i=0; i<n_sum; i++){
			int tmp_start = sum_pos_start[i];
			int* tmp_pos = sum_pos+tmp_start;
			int tmp_n = *(tmp_pos++);
			std::cout << "i = " << i << " n = " << tmp_n << ", ";
			for(int j=0; j<tmp_n; j++){
				std::cout << *tmp_pos++ << " ";
			}
			std::cout << "   sum_pos_start = " << tmp_start << " output = " << *tmp_pos++ << std::endl;
		}
	}

	void eval(CT* sum, CT* matrix){
		for(int i=0; i<n_sum; i++){
			int tmp_start = sum_pos_start[i];
			int* tmp_pos = sum_pos+tmp_start;
			int tmp_n = *(tmp_pos++);
			//std::cout << "i = " << i << " n = " << tmp_n << ", ";
			//std::cout << *tmp_pos << " " << sum[*tmp_pos] << " ";
			CT tmp = sum[*tmp_pos++];
			for(int j=1; j<tmp_n; j++){
				//std::cout << *tmp_pos << " " << sum[*tmp_pos] << " ";
				tmp += sum[*tmp_pos++];
			}
			matrix[*tmp_pos] = tmp;
			//std::cout << "sum_pos_start = " << tmp_start << " output = " << *tmp_pos << " " << tmp;
		}
	}
};

class CPUInstHom{
public:
    CPUInstHomCoef CPU_inst_hom_coef;
    CPUInstHomMon CPU_inst_hom_mon;
    CPUInstHomSum CPU_inst_hom_sum;

    int n_constant;
    int dim;
    int n_eq;
    int n_coef;
    int n_predictor;

    // Record timing for both CPU and GPU
	double timeSec_Path_CPU;
	double timeSec_Path_GPU;
	bool success_CPU;
	bool success_GPU;
	int n_step_CPU;
	int n_step_GPU;




    void init(MonSet* hom_monset, int n_monset, \
    		  int n_constant, int total_n_mon, int dim, int n_eq, int n_predictor, CT alpha);

	void init(PolySys& Target_Sys, PolySys& Start_Sys, int dim, int n_eq, int n_predictor, CT alpha);

    CPUInstHom(){
    	n_constant = 0;
    	dim = 0;
    	n_eq = 0;
    	n_coef = 0;
    	n_predictor = 0;
    	timeSec_Path_CPU = 0;
    	timeSec_Path_GPU = 0;
    	success_CPU = 0;
    	success_GPU = 0;
    	n_step_CPU = 0;
    	n_step_GPU = 0;
    }

    CPUInstHom(MonSet* hom_monset, int n_monset, int n_constant, int total_n_mon, int dim, int n_eq, int n_predictor, CT alpha)
	{
    	init(hom_monset, n_monset, n_constant, total_n_mon, dim, n_eq, n_predictor, alpha);
	}

    ~CPUInstHom(){
    }

    void print();

    void init_workspace(Workspace& workspace_cpu);

    void eval(Workspace& workspace_cpu, const CT* sol, const CT t, int reverse=0);

    void update_alpha();

};


#endif /* CPU_INSTRUCTION_EVAL_H_ */
