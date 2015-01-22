#ifndef __POLY_H__
#define __POLY_H__

#include <iostream>
#include <fstream>
#include "stdlib.h"
#include <string>
#include <cmath>
#include <vector>

#include "linklist.h"
//#include "job.h"
#include "dict.h"
#include "utilities.h"


/*typedef MonData<CT> MonNode;

typedef LinkList<MonNode*> PolyLink;

typedef LinkNode<MonNode*> PolyNode;*/


using namespace std;

/// Monomial class
/**
Read the monomial from the string.
PolyMon is used to construct polynomial equaion, PolyEq,
which is used to construct polynomial system, PolySys.
@sa PolyEq PolySys
*/
class PolyMon
{
public:
	/// coefficient of the monomial
	CT coef;
	/// derivative array of the monomial
	int n_var;
	/// position array of variables in the monomial
    int dim;
	int* pos;
	/// exponent array of variables in the monomial
	int* exp;

    int** pos_level;
    int* n_pos_level;
    int* pos_level_last;

    int n_level;


	PolyMon(){
        dim = 0;
		pos = NULL;
		exp = NULL;
	}

	PolyMon(int dim){
        this -> dim = dim;
		pos = NULL;
		exp = NULL;
	}


	~PolyMon(){
		delete[] pos;
		delete[] exp;
        //cout << "mon destructed" << endl;
	}

	/// Read monomial from certain part in equation string
	/**
	Coefficient is given.
	@param eq_string equation string
	@param pos_dict the dictionary of variables and their positions
	@param start start index of the monomial in equation string
	@param end end index of the monomial in equation string
	@param coef0 the coefficient of the monomail
	*/
	void read(const string& eq_string, VarDict& pos_dict, int start, int end, CT coef, int* max_deg);

	/// Read monomial from monomial string
	/**
	Coefficient is given.
	@param mon_string monomial string
	@param pos_dict the dictionary of variables and their positions
	@param coef0 the coefficient of the monomail
	*/
	void read(const string& mon_string, VarDict& pos_dict, CT coef, int* max_deg); 

	/// Read monomial from monomial string
	/**
	Read coefficient first and then variables.
	@param mon_string monomial string
	@param pos_dict the dictionary of variables and their positions
	*/
	void read(const string& mon_string, VarDict& pos_dict, int* max_deg);

	/// Use speel expanding to compute derivatives
	/**
	First compute forward product, 
	then compute backward and cross product together.
	@param[in] x_val values of variables
	@param[out] deri array of derivatives
	@return value of product of variables
	*/
	CT speel(const CT* x_val, CT* deri);

	/// Compute base of monomial
	/**
	@param[in] x_val values of variables
	@return value of base
	*/
	CT eval_base(const CT* x_val);

	/// Evaluate monomial
	/**
	@param x_val array of variables' values
	@return CT monomial value
	*/
	CT eval(const CT* x_val);

	/// Evaluate monomial and its derivatives
	/**
	@param[in] x_val array of variables' values
	@param[out] deri array of derivatives
	@return monomial value
	@sa eval_base() speel()
	*/
	CT eval(const CT* x_val, CT* deri);

	/// Print monomial
	/**
	Variable symbols are read from pos_dict.
	@param pos_dict the dictionary of variables and their positions
	*/
	void print(const string* pos_var);

    int memory_size(int factor_size);

    /// print level structure
    void print_level();
    int job_number_block(int start_level);

};

/// Polynomial equation class
/**
Read polynomial equation from string,
and build array of monomials, PolyMon.
PolyEq is used to construct polynomial system, PolySys.
@sa PolyMon PolySys
*/
class PolyEq{
public:
	/// Constant of equation
	CT constant;
	/// Number of monomials
	int n_mon;
	/// Number of monomials
	int dim;
	/// Monomial array of the equation
	vector<PolyMon*> mon;

    int level;
    int* job_number_level;

	/// Constructor 
	PolyEq(){
		constant.init(0.0,0.0);
		n_mon = 0;
		dim = 0;
	}

	/// Constructor 
	PolyEq(int dim){
		constant.init(0.0, 0.0);
		n_mon = 0;
		this -> dim = dim;
	}



	/// Destructor
	~PolyEq(){
        for(vector<PolyMon*>::iterator it=mon.begin(); it<mon.end(); it++){
            delete *it;
        }
        //cout << "eq destructed" << endl;
	}

	/// Read equaiton from a string
	/**
	First, indentify monomials position.
	Then, construct an array of monomials, use PolyMon::read().
	@param eq_string equation string
	@param pos_dict the dictionary of variables and their positions
	@sa PolyMon::read()
	*/
	void read(const string& eq_string, VarDict& pos_dict, int* max_deg);

	/// Read equaiton from certain part of a string
	/**
	First, indentify monomials position.
	Then, construct an array of monomials, use PolyMon::read().
	@param eq_string equation string
	@param pos_dict the dictionary of variables and their positions
	@param start the start position of equation in the string 
	@param end the end position of equation in the string
	@sa PolyMon::read()
	*/
	void read(const string& eq_string, VarDict& pos_dict, int start, int end, int* max_deg);

	/// Print equation
	/**
	Print monomial by monomial in the equation.
	Constant comes the last.
	@param pos_dict the dictionary of variables and their positions
	*/
	void print(const string* pos_var);

	/// Evaluate equation
	/**
	@param x_val CT array of variables' values
	@return CT equation value
	*/
	CT eval(const CT* x_val);
	
	/// Evaluate equation and its derivatives
	/**
	Evalue monomials' value and derivative by PolyMon::eval(),
	and add them up.
	@param[in] x_val array of variables' values
	@param[out] deri array of derivatives
	@return equation value
	*/
	CT eval(const CT* x_val, CT* deri);

    int memory_size(int factor_size);

    void print_level();

    int workspace_size_block(int start_level, int factor_size);
    int job_number_block(int start_level);

};


/// Polynomial system class
/**
Read polynomial system from string or string arrays,
and build array of polynomial equations, PolyEq.
@sa PolyMon, PolyEq.
*/
class PolySys{
public:
	/// Number of Equations
	int n_eq;
	/// Dimension
	int dim;
    string* pos_var;
	/// Array of polynomial equations
	vector<PolyEq*> eq;
	
	PolyEq* eq_space;

	int* max_deg;

    int* job_number_level;
    int level;

	/// Constructor
	PolySys(){
		n_eq = 0;
		max_deg = NULL;
		pos_var = NULL;
	}

	/// Destructor
	~PolySys(){
		//delete eq_space;
        //delete pos_var;
        delete max_deg;
		cout << "sys destructed" << endl;
	}

	/// Read polynomial system from string array
	/**
	Read equation from string array, sys_string, using PolyEq::read().
	@param sys_string polynomial system string
	@param n number of equations or number of strings in string array sys_string
	@param pos_dict the dictionary of variables and their positions
	@sa PolyEq::read()
	*/
	void read(const string* sys_string, int n_eq, VarDict& pos_dict);

	/// Read polynomial system from string
	/**
	First split the string by ';', then use PolyEq::read() to read each equation.
	@param sys_string polynomial system string
	@param pos_dict the dictionary of variables and their positions
	@sa PolyEq::read()
	*/
	void read(const string& sys_string, VarDict& pos_dict);

    ///Read polynomial system from file
    /**
    Read dimension on the first line and then system.
    @param filename file name of polynomial system. File requirement: first line is dimension, equations are split by ';'
    @param pos_dict position dictionary for variables. If it is empty, the positions will be created by the order of variables appearance first time in the file.
    @sa read_file(ifstream& myfile, Dict& pos_dict)
    */
    void read_file(const string& file_name);

    ///Read polynomial system from file
    /**
    Read dimension on the first line and then system.
    @param myfile file stream of polynomial system. File requirement: first line is dimension, equations are split by ';'
    @param pos_dict position dictionary for variables. If it is empty, the positions will be created by the order of variables appearance first time in the file.
    @sa read_file(string file_name, Dict& pos_dict)
    */
    void read_file(ifstream& myfile, VarDict& pos_dict);

	/// Evaluate polynomial system
	/**
	@param x_val CT array of variables' values
	@return CT array of system value
	*/
	CT* eval(const CT* x_val);
	
	/// Evaluate equation and its derivative
	/**
	Evalue monomials' value and derivative by PolyMon::eval(),
	and add them up.
	@param[in] x_val array of variables' values
	@param[out] deri array of derivatives
	@return equation value
	*/
	CT* eval(const CT* x_val, CT** deri);

	void eval(const CT* x_val, CT* f_val,  CT** deri_val);

	/// Print polynomial system
	/**
	Use PolyEq::print() to print each equation in the system.
	@param pos_dict the dictionary of variables and their positions
	@return void
	@sa PolyEq::print() PolyMon::print()
	*/
	void print();
    
    void gpu_mon(int& dim, int& level, int& workspace_size, int*& workspace_level,
                 int*& n_mon_level, int& pos_size, unsigned short*& pos, int*& pos_level,
                 int& sum_level, int*& n_sum_level,
                 int& total_n_sum, int& sum_array_size, int*& sum_start, int*& sum_array);

/*	void factor(Job_Poly* jobs);

	void factor1(Job_Poly* jobs, int start_level=1);

	void factor_block(Job_Poly_Block* jobs, int start_level, long Cap);*/

    int workspace_size_block(int start_level, int factor_size);

    int memory_size(int factor_size);

    void job_number();

    void print_level();

    int job_number_block(int start_level);
};

class PolySysHom{
public:
	PolySys* start_sys;
	PolySys* target_sys;
	int dim;
	int n_start_sol;
	CT* start_sols;

	PolySysHom(PolySys* start_sys, PolySys* target_sys){
		if(start_sys->dim != target_sys->dim){
			std::cout << "start system and end system " << std::endl;
		}
		else{
			this->start_sys = start_sys;
			this->target_sys = target_sys;
			dim = start_sys->dim;
		}
	}

	void print(){
		std::cout << "Start System : " << std::endl;
		start_sys->print();
		std::cout << "Target System : " << std::endl;
		target_sys->print();
	}
};

class int_idx{
  public:
    int eq_idx;
    int mon_idx;
    int_idx(int i, int j){
        eq_idx = i;
        mon_idx = j;
    }
};

class PolySol{
public:
	int dim;
	// solution number
	int idx;
	// multiplicity
	int m;

	CT t;

	CT* sol;
	// error
	T1 err;
	// conditional number
	T1 rco;
	// residual
	T1 res;

	void init(ifstream& myfile, int dim);

	PolySol(){
	}

	PolySol(ifstream& myfile, int dim){
		init(myfile, dim);
	}

	~PolySol(){
		delete[] sol;
		//std::cout << "PolySol Deleted" << std::endl;
	}

	void print(){
		std::cout << "Solution " << idx << ":" << std::endl;
		std::cout << "t : " << t;
		for(int i=0; i<dim; i++){
			std::cout << i << " : " << sol[i];
		}
		std::cout << "== err : " << err \
			      << " = rco : " << rco \
			      << " = res : " << res \
			      << " ==" << std::endl;
	}

	CT* get_sol(){
		CT* sol_tmp = new CT[dim];
		for(int i=0; i<dim; i++){
			sol_tmp[i] = sol[i];
		}
		return sol_tmp;
	}
};

class PolySolSet{
public:
	int n_sol;
	int dim;
	PolySol* sols;

	PolySolSet(){
		n_sol = 0;
		dim = 0;
		sols = NULL;
	}

	void init(ifstream& myfile);

	PolySolSet(ifstream& myfile){
		init(myfile);
	}

	~PolySolSet(){
		delete[] sols;
		sols = NULL;
		//std::cout << "PolySol Deleted" << std::endl;
	}

	void print(){
		std::cout << "dim   = " << dim << std::endl\
				  << "n_sol = " << n_sol << std::endl;

		for(int i=0; i<n_sol; i++){
			sols[i].print();
		}

	}

	CT* get_sol(int idx){
		return sols[idx].get_sol();
	}
};

#endif
