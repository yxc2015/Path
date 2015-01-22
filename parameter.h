/*
 * parameter.h
 *
 *  Created on: Dec 25, 2014
 *      Author: yxc
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

class Parameter{
public:
	int n_predictor;

	int max_step;

	int max_it;
	double max_delta_t;
	double min_delta_t;
	double err_max_res;
	double err_max_delta_x;
	double err_max_first_delta_x;
	double err_min_round_off;

	Parameter(int n_predictor, int max_step, int max_it, \
			double max_delta_t, double min_delta_t, \
			double err_max_res, double err_max_delta_x, \
			double err_max_first_delta_x, double err_min_round_off){
		this->n_predictor = n_predictor;
		this->max_step = max_step;
		this->max_it = max_it;
		this->max_delta_t = max_delta_t;
		this->min_delta_t = min_delta_t;
		this->err_max_res = err_max_res;
		this->err_max_delta_x = err_max_delta_x;
		this->err_max_first_delta_x = err_max_first_delta_x;
		this->err_min_round_off = err_min_round_off;

		std::cout << "path_parameter.err_min_round_off = " << this->err_min_round_off << std::endl;
	}
};


#endif /* PARAMETER_H_ */
