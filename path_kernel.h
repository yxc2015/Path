/*
 * eval_kernel.h
 *
 *  Created on: Dec 3, 2014
 *      Author: yxc
 */

#ifndef EVAL_KERNEL_H_
#define EVAL_KERNEL_H_

#include <iostream>

#include "DefineType.h"
#include "eval_host.h"

#define DEBUG 0

int GPU_Eval(const CPUInstHom& hom, CT* cpu_sol0, CT cpu_t, CT*& gpu_workspace, CT*& gpu_matrix, \
		     int n_predictor=1, int n_sys = 1);

int GPU_Predict(const CPUInstHom& hom, CT*& x_gpu, int n_predictor, CT cpu_t, int n_sys = 1);

int GPU_MGS(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q, CT*& matrix_gpu_r, int n_predictor, CT* V, int n_sys=1);

int GPU_MGS_Large(const CPUInstHom& hom, CT*& sol_gpu, CT*& matrix_gpu_q, CT*& matrix_gpu_r, int n_predictor, CT* V, int n_sys=1);

bool GPU_Newton(CPUInstHom& hom, Parameter path_parameter, CT* cpu_sol0, CT cpu_t, CT*& x_new,int n_sys=1);

bool GPU_Path(CPUInstHom& hom, Parameter path_parameter, CT* cpu_sol0, CT cpu_t, CT*& x_gpu, int inverse=0, int n_sys=1);

#endif /* EVAL_KERNEL_H_ */
