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
