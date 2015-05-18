/*
 * init_test.h
 *
 *  Created on: Feb 8, 2015
 *      Author: yxc
 */

#ifndef INIT_TEST_H_
#define INIT_TEST_H_

#include "families.h"
#include "eval_host.h"

bool init_test(PolySys& Target_Sys, PolySys& Start_Sys, PolySolSet& sol_set, int dim, int& n_eq, \
		       CT*& sol0, CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, int test, int n_predictor, int sys_type = 2);

void init_cpu_inst_workspace(PolySys& Target_Sys, PolySys& Start_Sys, \
		                     int dim, int n_eq, int n_predictor, \
	                         CPUInstHom& cpu_inst_hom, Workspace& workspace_cpu, \
	                         int test);

bool read_homotopy_file(PolySys& Target_Sys, PolySys& Start_Sys,\
		int dim, int& n_eq, \
		string Start_Sys_filename, string Target_Sys_file_name, \
		PolySolSet* sol_set=NULL);

#endif /* INIT_TEST_H_ */
