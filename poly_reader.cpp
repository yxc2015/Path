/** @file */

#include <iostream>
#include <fstream>

#include "DefineType.h"
#include "utilities.h"


int main(){
    string a = "+(-1.44377044086739E-01 + 9.89522748167406E-01*i)";
    int loc = 0;
    CT v = get_coef_complex(a, loc);
    std::cout << v << std::endl;
    CT h = v;
    CT w = h/v;
    w.init(1.0,2.0);
    std::cout << v;
    std::cout << h;
    std::cout << w;
    std::cout << endl;

    ifstream openfile("cyclic8_start");

    if(openfile.is_open()){
    	int dim;
    	CT test_var;
    	openfile.ignore(10,'(');
    	openfile >> test_var;
    	std::cout << test_var << std::endl;
    	//openfile.ignore(10,'*');
    	string a;
    	openfile >> a;
    	std::cout << a << std::endl;
    }
    else{

    }


    return 0;
}
