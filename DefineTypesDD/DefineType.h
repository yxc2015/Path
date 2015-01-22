/* This file defines the precision level on host and card.
 * Type T is for the card and is either double, gdd_real, or gqd_real.
 * The corresponding type T1 on the host is double, dd_real, or qd_real.
 * The definition of the precision level is set at compile time with
 * the gcc flag "-D precision=d" for double precision
 *              "-D precision=dd" for double double precision, and
 *              "-D precision=qd" for quad double precision. */

#ifndef __DEFINE_TYPE_DD_H__
#define __DEFINE_TYPE_DD_H__

#include <qd/qd_real.h>
#include "../complexH.h"
#include "../complex.h"

typedef gdd_real T;
typedef dd_real T1;

#define ERR 1E-25

#define CT complexH<dd_real>
#define GT complex<gdd_real>

inline dd_real read_number(const char* number){
	return dd_real(number);
}

#define shmemsize 256

#define matrix_block_row 64
#define matrix_block_pivot_col 1
#define matrix_block_reduce_col 2

// Parameters
#define N_PREDICTOR           4

#define MAX_STEP              1500
#define MAX_DELTA_T           1E-1
#define MIN_DELTA_T           1E-20

#define MAX_IT                5
#define ERR_MAX_RES           1E-1
#define ERR_MAX_DELTA_X       1E-1
#define ERR_MAX_FIRST_DELTA_X 1E-2
#define ERR_MIN_ROUND_OFF     1E-30
#endif
