/*
 * Parameter_Header.h
 *
 *  Created on: Feb 1, 2015
 *      Author: yxc
 */

#ifndef PARAMETER_HEADER_H_
#define PARAMETER_HEADER_H_

#define ERR 1E-10

#define maxrounds 128

#define max_array_size 2000

#define shmemsize 512

#define BS_QR 256

#define BS_QR_Back 256

// QR Reduce Parameters
#define matrix_block_row 32
#define matrix_block_pivot_col 4
#define matrix_block_reduce_col 4

// Parameters
#define N_PREDICTOR           4

#define MAX_STEP              1000
#define MAX_DELTA_T           1E-1
#define MIN_DELTA_T           1E-20

#define MAX_IT                3
#define ERR_MAX_RES           1E-2
#define ERR_MAX_DELTA_X       1E-1
#define ERR_MAX_FIRST_DELTA_X 1E-2
#define ERR_MIN_ROUND_OFF     1E-15


#endif /* PARAMETER_HEADER_H_ */