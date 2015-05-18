/*
 * eval_test.h
 *
 *  Created on: Feb 8, 2015
 *      Author: yxc
 */

#ifndef EVAL_TEST_H_
#define EVAL_TEST_H_

#include "eval_host.h"
#include "path_gpu.h"
#include "err_check.h"

T1 eval_test(const CPUInstHom& cpu_inst_hom, CT* host_sol0, CT t, const CT* cpu_workspace, const CT* cpu_matrix);

T1 eval_test_classic(Workspace& workspace_cpu, CPUInstHom& cpu_inst_hom, CT* sol0, CT t, PolySys& Target_Sys, int n_eq, int dim, int n_path=-1);

#endif /* EVAL_TEST_H_ */
