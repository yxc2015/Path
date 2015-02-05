/*
 * eval_host.cpp
 *
 *  Created on: Dec 6, 2014
 *      Author: yxc
 */

#include "eval_host.h"

void polysys_mon_set(const PolySys& Target_Sys, MonIdxSet* mon_set, bool sys_idx){
	PolyEq* tmp_eq= Target_Sys.eq_space;

	int mon_idx = 0;
    for(int i=0; i<Target_Sys.n_eq; i++){
    	if(tmp_eq->constant.real != 0.0 || tmp_eq->constant.imag != 0.0){
    		mon_set[mon_idx] = MonIdxSet(0, NULL, i, 0, sys_idx, tmp_eq->constant);
    		//std::cout << "constant = " << tmp_eq->constant << endl;
    		mon_idx++;
    	}
    	for(int j=0; j<tmp_eq->n_mon; j++){
    		PolyMon* tmp_mon = tmp_eq->mon[j];
    		mon_set[mon_idx] = MonIdxSet(tmp_mon->n_var, tmp_mon->pos, i, j, sys_idx, tmp_mon->coef);
    		mon_idx++;
    	}
    	tmp_eq++;
    }
}

MonIdxSet* polysyshom_monidxset(PolySys& Target_Sys, PolySys& Start_Sys, int& total_n_mon){
	total_n_mon = 0;
	PolyEq* tmp_eq= Target_Sys.eq_space;
    for(int i=0; i<Target_Sys.n_eq; i++){
    	total_n_mon += tmp_eq->n_mon;
    	if(tmp_eq->constant.real != 0.0 || tmp_eq->constant.imag != 0.0){
    		total_n_mon++;
    	}
    	tmp_eq++;
    }

    int total_n_mon_start = total_n_mon;

	tmp_eq= Start_Sys.eq_space;
    for(int i=0; i<Start_Sys.n_eq; i++){
    	total_n_mon += tmp_eq->n_mon;
    	if(tmp_eq->constant.real != 0.0 || tmp_eq->constant.imag != 0.0){
    		total_n_mon++;
    	}
    	tmp_eq++;
    }

    std::cout << "total_n_mon = " << total_n_mon << std::endl;

    MonIdxSet* mon_set = new MonIdxSet[total_n_mon];

    polysys_mon_set(Target_Sys, mon_set, 0);

    polysys_mon_set(Start_Sys, mon_set+total_n_mon_start, 1);

    for(int i=0; i<total_n_mon; i++){
    	mon_set[i].sorted();
    }

    std::cout << "start sort" << std::endl;
    std::sort(mon_set, mon_set+total_n_mon);
    std::cout << "end sort" << std::endl;

    /*for(int i=0; i<total_n_mon; i++){
    	mon_set[i].print();
    }*/

    return mon_set;
}


MonSet* polysyshom_monset(int total_n_mon, MonIdxSet* mons,
		                  int& n_constant, int& hom_n_mon, int& n_monset)
{
	std::cout << "total_n_mon = " << total_n_mon << std::endl;
	n_monset = 1;

    // Mark new location
	int* new_type = new int[total_n_mon];
    new_type[0] = 0;


    // Check monomial type
    MonIdxSet tmp_mon = mons[0];
	//std::cout << 0 << std::endl;
	//mons[0].print();

    for(int i=1; i<total_n_mon; i++){
    	//std::cout << i << std::endl;
    	//mons[i].print();
    	if(mons[i] == tmp_mon){
    		new_type[i] = 0;
    	}
    	else{
    		new_type[i-1] = 1;
    		n_monset++;
    		tmp_mon = mons[i];
    	}
    }
    new_type[total_n_mon-1] = 1;

	std::cout << "n_mon_type = " << n_monset << std::endl;

    int* n_mons = new int[n_monset];
    int tmp_n_mon = 0;
    int tmp_eq_idx = -1;
    int monset_idx = 0;
    for(int i=0; i<total_n_mon; i++){
    	if(tmp_eq_idx != mons[i].get_eq_idx()){
    		tmp_n_mon++;
    		tmp_eq_idx = mons[i].get_eq_idx();
    	}
    	if(new_type[i] == 1){
    		n_mons[monset_idx] = tmp_n_mon;
    		//std::cout << n_mons[monset_idx] << ", ";
    		tmp_n_mon = 0;
    		tmp_eq_idx = -1;
    		monset_idx++;
    	}
    }
    //std::cout << std::endl;

    MonSet* hom_monset = new MonSet[n_monset];
    int mon_idx = 0;
    for(int i=0; i<n_monset; i++){
    	hom_monset[i].copy_pos(mons[mon_idx]);
    	//int* tmp_eq_idx = new int[n_mons[i]];
    	//CT* tmp_coef = new CT[2*n_mons[i]];
    	EqIdxCoef* tmp_eq_idx_coef = new EqIdxCoef[n_mons[i]];
		//std::cout << "n_mons "<< i << " " << n_mons[i]<< std::endl;
    	for(int j=0; j<n_mons[i]; j++){
    		// merge by eq_idx
    		if(mons[mon_idx].get_sys_idx() == 0){
        		int tmp_eq_idx = mons[mon_idx].get_eq_idx();
        		mon_idx++;
        		if(mons[mon_idx].get_sys_idx() == 1 && tmp_eq_idx == mons[mon_idx].get_eq_idx()){
        			//std::cout << mons[mon_idx-1].get_coef();
        		    //std::cout << mons[mon_idx].get_coef();
        		    //std::cout << std::endl;
	        		tmp_eq_idx_coef[j] = EqIdxCoef(tmp_eq_idx, mons[mon_idx-1].get_coef(), mons[mon_idx].get_coef());
					mon_idx++;
				}
				else{
        		    //std::cout << 0 << std::endl;
        		    //std::cout << mons[mon_idx-1].get_coef();
        		    //std::cout << std::endl;
	        		tmp_eq_idx_coef[j] = EqIdxCoef(tmp_eq_idx, mons[mon_idx-1].get_coef(), 0);
				}
    		}
    		else{
    		    std::cout << 1 << std::endl;
    		    std::cout << mons[mon_idx].get_coef();
    		    std::cout << std::endl;
        		tmp_eq_idx_coef[j] = EqIdxCoef(tmp_eq_idx, mons[mon_idx].get_coef(), 1);

    		}
    	}
        hom_monset[i].update_eq_idx(n_mons[i],tmp_eq_idx_coef);
    }
    std::cout << "Finished" << std::endl;

    // Get number of constants
    if(hom_monset[0].get_n() == 0){
    	n_constant = hom_monset[0].get_n_mon();
    }

    hom_n_mon = 0;
	for(int i=0; i<n_monset; i++){
		//std::cout << hom_monset[i];
		hom_n_mon += hom_monset[i].get_n_mon();
	}

	return hom_monset;
}

MonSet* hom_monset_generator(PolySys& Target_Sys, PolySys& Start_Sys, int& n_monset, int& n_constant, int& total_n_mon){
	int hom_n_mon;
    MonIdxSet* mons = polysyshom_monidxset(Target_Sys, Start_Sys, hom_n_mon);
    MonSet* hom_monset = polysyshom_monset(hom_n_mon, mons, n_constant, total_n_mon, n_monset);
    return hom_monset;
}


void CPUInstHomCoef::init(MonSet* hom_monset, int total_n_mon, int n_monset, int n_constant, CT alpha){
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

void CPUInstHomCoef::print(){
	//Print coefficient
	for(int i=0; i<n_coef; i++){
	std::cout << i << std::endl
	          << coef_orig[2*i]
	          << coef_orig[2*i+1]<< std::endl;
	}
}

void CPUInstHomCoef::eval(const CT t, CT* coef, int reverse){
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


void CPUInstHom::init(MonSet* hom_monset, int n_monset, \
		  int n_constant, int total_n_mon, int dim, int n_eq, int n_predictor, CT alpha){
	this->n_constant = n_constant;
	this->dim = dim;
	this->n_eq = n_eq;
	this->n_predictor = n_predictor;

	std::cout << "Generating CPU Instruction ..." << std::endl;
	std::cout << "           Coef Instruction ..." << std::endl;
	CPU_inst_hom_coef.init(hom_monset, total_n_mon, n_monset, n_constant, alpha);
	//CPU_inst_hom_coef.print();
	std::cout << "           Mon Instruction ..." << std::endl;
	CPU_inst_hom_mon.init(hom_monset, n_monset, total_n_mon, n_constant);
	//CPU_inst_hom_mon.print();
	std::cout << "           Sum Instruction ..." << std::endl;
	CPU_inst_hom_sum.init(hom_monset, n_monset, CPU_inst_hom_mon.mon_pos_start, dim, n_eq, n_constant);
	//CPU_inst_hom_sum.print();

	std::cout << "Generating CPU Instruction Finished" << std::endl;

	this->n_coef = CPU_inst_hom_coef.n_coef;
	//CPU_inst_hom_mon.print();
	CPU_inst_hom_block.init(CPU_inst_hom_mon, warp_size);
	//CPU_inst_hom_block.print();
	CPU_inst_hom_sum_block.init(hom_monset, n_monset, CPU_inst_hom_mon.mon_pos_start, dim, n_eq, n_constant,\
						  CPU_inst_hom_mon.n_mon_level[0],CPU_inst_hom_block.mon_pos_start_block);
	//CPU_inst_hom_sum_block.print();
}


void CPUInstHom::init(PolySys& Target_Sys, PolySys& Start_Sys, int dim, int n_eq, int n_predictor, CT alpha){
    int n_constant;
    int total_n_mon;
    int n_monset;
    int n_mon;
    MonSet* hom_monset = hom_monset_generator(Target_Sys, Start_Sys, n_monset, n_constant, total_n_mon);

    std::cout << "n_constant  = " << n_constant << std::endl;
    std::cout << "total_n_mon = " << total_n_mon << std::endl;
    std::cout << "n_monset    = " << n_monset << std::endl;

    init(hom_monset, n_monset, n_constant, total_n_mon, dim, n_eq, n_predictor, alpha);
}

void CPUInstHom::print(){
	std::cout << "*************** Coef Instruction ********************" << std::endl;
	CPU_inst_hom_coef.print();
	std::cout << "*************** Mon Instruction ********************" << std::endl;
	CPU_inst_hom_mon.print();
	std::cout << "*************** Sum Instruction ********************" << std::endl;
	CPU_inst_hom_sum.print();
}


void CPUInstHom::init_workspace(Workspace& workspace_cpu){
	int coef_size = CPU_inst_hom_coef.n_coef;
	int workspace_size = coef_size + CPU_inst_hom_mon.mon_pos_size;
	workspace_cpu.init(workspace_size, coef_size, n_constant, n_eq, dim, n_predictor);
	std::cout << "workspace_size = " << workspace_size << std::endl;
	std::cout << "coef_size = " << coef_size << std::endl;
}

void CPUInstHom::eval(Workspace& workspace_cpu, const CT* sol, const CT t, int reverse){
    struct timeval start, end;
    long seconds, useconds;
    double mtime;

	//CPU_inst_hom_coef.print();
	//begin = clock();
	CPU_inst_hom_coef.eval(t, workspace_cpu.coef, reverse);
	//end = clock();
	//timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
	//std::cout << "CPU Eval Coef time " <<  timeSec << std::endl;

	//begin = clock();
	CPU_inst_hom_mon.eval(sol, workspace_cpu.mon, workspace_cpu.coef);
	//end = clock();
	//timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
	//std::cout << "CPU Eval Mon time " <<  timeSec << std::endl;
	/*std::cout << "----- Workspace ----" << std::endl;
	for(int i=0; i<50; i++){
		std::cout << i << " " << workspace_cpu.sum[i];
	}*/

	gettimeofday(&start, NULL);
	CPU_inst_hom_sum.eval(workspace_cpu.sum, workspace_cpu.matrix);
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

	std::cout << "CPU Eval Sum time " <<  mtime << std::endl;
}

void CPUInstHom::update_alpha(){
	CPU_inst_hom_coef.update_alpha();
}

void CPUInstHomCoef::update_alpha(){
	int r = rand();
	T1 tmp = T1(r);
	alpha = CT(sin(tmp),cos(tmp));
}



void CPUInstHomMon::init(MonSet* hom_monset, int n_monset, int total_n_mon, int n_constant)
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

void CPUInstHomMon::eval(const CT* sol, CT* mon, CT* coef){
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

void CPUInstHomMon::print(){
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

void CPUInstHomMonBlock::init(CPUInstHomMon& orig, int BS){
	n_mon = orig.n_mon;
	this->BS = BS;
	int mon_0 = 0;
	for(int i=0; i<n_mon; i++){
		if(orig.mon_pos[orig.mon_pos_start[i]]==1){
			mon_0++;
		}
	}
	n_mon -= mon_0;
	NB = (n_mon-1)/BS + 1;
	max_var_block = new unsigned short[NB];
	for(int i=0; i<n_mon; i++){
		int bidx = i/BS;
		int tidx = i - bidx*BS;
		unsigned short n_var = orig.mon_pos[orig.mon_pos_start[i+mon_0]];
		if(tidx == 0){
			max_var_block[bidx] = n_var;
		}
		else{
			if(n_var > max_var_block[bidx]){
				max_var_block[bidx] = n_var;
			}
		}
	}

	mon_pos_start_block = new int[NB];

	mon_pos_block_size = 0;
	for(int i=0; i<NB; i++){
		mon_pos_start_block[i] = mon_pos_block_size;
		mon_pos_block_size += BS*(max_var_block[i]+1);
		//std::cout << "max_var_block[" << i << "] = " << max_var_block[i] \
				  << " start = " << mon_pos_start_block[i] << std::endl;
	}
	std::cout << "mon_pos_block_size = " << mon_pos_block_size << std::endl;

	mon_pos_block = new unsigned short[mon_pos_block_size];

	unsigned short* tmp_mon_pos = orig.mon_pos+orig.mon_pos_start[mon_0];
	for(int i=0; i<NB; i++){
		unsigned short* tmp_mon_pos_block = mon_pos_block + mon_pos_start_block[i];
		for(int j=0; j<BS; j++){
			int mon_idx = i*BS+j;
			if(mon_idx<n_mon){
				unsigned short n_var = *tmp_mon_pos++;
				//std::cout << i << " " << j << " " << n_var << std::endl;
				tmp_mon_pos_block[j] = n_var;
				for(int k=0; k<n_var; k++){
					//std::cout << i << " " << j << " " << k << std::endl;
					tmp_mon_pos_block[(k+1)*BS+j] = *tmp_mon_pos++;
				}
			}
		}
	}
}

void CPUInstHomMonBlock::print(){
	std::cout << "BS = " << BS << std::endl;
	std::cout << "NB = " << NB << std::endl;
	for(int i=0; i<NB; i++){
		std::cout << "BS " << i << " n_var = " << max_var_block[i] \
				  << " start = " << mon_pos_start_block[i] << std::endl;
		unsigned short* tmp_mon_pos_block = mon_pos_block + mon_pos_start_block[i];
		for(int j=0; j<BS; j++){
			int mon_idx = i*BS+j;
			if(mon_idx<n_mon){
				unsigned short n_var = tmp_mon_pos_block[j];
				std::cout << mon_idx << " n_var=" << n_var;
				for(int k=0; k<n_var; k++){
					std::cout << " " << tmp_mon_pos_block[(k+1)*BS+j];
				}
				std::cout << std::endl;
			}
		}
	}
}



void CPUInstHomSumBlock::init(MonSet* hom_monset, int n_monset, const int* mon_pos_start,
		             int dim, int n_eq, int n_constant, int n_mon0, int* mon_pos_start_block)
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
			int n_var = tmp_hom_monset->get_n();
			int tmp_pos;
			int bidx;
			int tidx;
			int tmp_start_block;
			int tmp_mon_idx;
			if(n_var < 2){
				tmp_pos = mon_pos_start[mon_idx]+n_constant;
			}
			else{
				tmp_mon_idx = mon_idx - n_mon0;
				bidx = tmp_mon_idx/warp_size;
				tidx = tmp_mon_idx - bidx*warp_size;
				tmp_start_block = mon_pos_start_block[bidx];
				//std::cout << "mon_idx = " << mon_idx \
						  << " tmp_mon_idx = " << tmp_mon_idx \
						  << " bidx = " << bidx \
						  << " tidx = " << tidx\
						  << " mon_pos_start_block[bidx] = "\
						  << mon_pos_start_block[bidx] << std::endl;
				tmp_start_block = tmp_start_block + n_mon0*2+n_constant;
				tmp_pos = tmp_start_block + tidx;
			}
			int tmp_eq_idx = tmp_hom_monset->get_eq_idx(j);
			// Value
			sum_pos[sum_pos_start_matrix[tmp_eq_idx][dim]] = tmp_pos;
			tmp_pos++;
			sum_pos_start_matrix[tmp_eq_idx][dim]++;
			n_sums[tmp_eq_idx][dim] += 1;
			// Derivative
			for(int k=0; k<n_var; k++){
				if(n_var < 2){
					/*std::cout << "   mon pos=" \
							  << tmp_pos;
					std::cout << " tmp_eq_idx=" << tmp_eq_idx;
					std::cout << " get_pos(k)=" << tmp_hom_monset->get_pos(k) << std::endl;*/
					sum_pos[sum_pos_start_matrix[tmp_eq_idx][tmp_hom_monset->get_pos(k)]] = tmp_pos;
					tmp_pos++;
				}
				else{
					/*std::cout << "   mon pos=" \
							  << tmp_start_block+(k+1)*warp_size+tidx \
							  << " tmp_start_block=" <<tmp_start_block ;
					std::cout << " tmp_eq_idx=" << tmp_eq_idx;
					std::cout << " get_pos(k)=" << tmp_hom_monset->get_pos(k) << std::endl;*/
					sum_pos[sum_pos_start_matrix[tmp_eq_idx][tmp_hom_monset->get_pos(k)]] = \
							tmp_start_block+(k+1)*warp_size+tidx;
				}
				sum_pos_start_matrix[tmp_eq_idx][tmp_hom_monset->get_pos(k)]++;
			}
			mon_idx++;
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

void CPUInstHomSumBlock::eval(CT* sum, CT* matrix){
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

void CPUInstHomSumBlock::print(){
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

void CPUInstHomSum::init(MonSet* hom_monset, int n_monset, const int* mon_pos_start, \
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

void CPUInstHomSum::eval(CT* sum, CT* matrix){
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

void CPUInstHomSum::print(){
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
