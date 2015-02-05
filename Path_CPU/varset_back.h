#include <iostream>
#include<algorithm>

#include "DefineType.h"

class VarSet{
public:
	int n;
	VarSet* elements;

	void init(){
		this->n = 0;
		elements = NULL;

	}
	void init(const int n){
		this->n = n;
		elements = NULL;
	}

	void init(const int n, const VarSet* elements){
		this->n = n;
		if(elements != NULL){
			this->elements = new VarSet[n];
			for(int i=0; i<n; i++){
				this->elements[i] = elements[i];
			}
		}
		else{
			this->elements = NULL;
		}
	};

	VarSet(){
		init();
	}

	VarSet(int n){
		init(n);
	}

	VarSet(const int n, const VarSet* elements){
		init(n, elements);
	}

	VarSet(const VarSet& original){
		init(original.n, original.elements);
	}

	VarSet& operator= (const VarSet& original){
		init(original.n, original.elements);
		return *this;
	}

	bool operator < (const VarSet& that) const{
		if(this->n < that.n){
			return true;
		}
		else if(this->n > that.n){
			return false;
		}
		else if(this->n > 0){
			// Check if one is empty
			if(this->elements == NULL && that.elements == NULL){
				return false;
			}
			else if(this->elements == NULL) {
				std::cout << "Not on the same level" << std::endl;
				return false;
			}
			else if(that.elements == NULL) {
				std::cout << "Not on the same level" << std::endl;
				return false;
			}

			// Compare each element
			for(int i=0; i<n; i++){
				if(this->elements[i] < that.elements[i]){
					return true;
				}
				else if(this->elements[i] > that.elements[i]){
					return false;
				}
			}
			return false;
		}
		else{
			return false;
		}
	}

	bool operator > (const VarSet& that) const{
		if(this->n > that.n){
			return true;
		}
		else if(this->n < that.n){
			return false;
		}
		else if(this->n > 0){
			// Check if one is empty
			if(this->elements == NULL && that.elements == NULL){
				return false;
			}
			else if(this->elements == NULL) {
				std::cout << "Not on the same level" << std::endl;
				return false;
			}
			else if(that.elements == NULL) {
				std::cout << "Not on the same level" << std::endl;
				return false;
			}

			for(int i=0; i<n; i++){
				if(this->elements[i] > that.elements[i]){
					return true;
				}
				else if(this->elements[i] < that.elements[i]){
					return false;
				}
			}
			return false;
		}
		else{
			return false;
		}
	}

	~VarSet(){
		delete[] elements;
		elements = NULL;
	}

	void print(int level = 0){
		if(elements != NULL){
			std::cout << std::endl;
			for(int i=0; i<level; i++){
				std::cout << "   ";
			}
			std::cout << "l"<< level << " n" << n << " : ";
			for(int i=0; i<n; i++){
				elements[i].print(level+1);
			}
		}
		else{
			std::cout << n << ", ";
		}
		if(level == 0){
			std::cout << std::endl;
		}
	}

	void sorted(){
		if(elements != NULL){
			for(int i=0; i<n; i++){
				elements[i].sorted();
			}
			std::sort(elements, elements+n);
		}
	}
};

class IntSet{
	int n;
	int* elements;

	void init(const int n, const int* elements){
		this->n = n;
		this->elements = new int[n];
		for(int i=0; i<n; i++){
			this->elements[i] = elements[i];
		}
	}

public:
	IntSet(){
		n = 0;
		elements = NULL;
	};

	IntSet(const int n, const int* elements){
		init(n, elements);
	}

	IntSet& operator= (const IntSet& original){
		init(original.n, original.elements);
		return *this;
	}

	~IntSet(){
		delete[] elements;
		elements = NULL;
	}

	int get_n(){
		return n;
	}

	int get_element(int i){
		return elements[i];
	}

	void update(int n, int* elements){
		this->n = n;
		this->elements = elements;
	}

	bool operator < (const IntSet& that) const{
		if(this->n < that.n){
			return true;
		}
		else if(this->n > that.n){
			return false;
		}
		else if(this->n > 0){
			// Check if one is empty
			if(this->elements == NULL || that.elements == NULL){
				std::cout << "At least, one set is empty" << std::endl;
				return false;
			}

			// Compare each element
			for(int i=0; i<n; i++){
				if(this->elements[i] < that.elements[i]){
					return true;
				}
				else if(this->elements[i] > that.elements[i]){
					return false;
				}
			}

			return false;
		}
		else{
			return false;
		}
	}

	bool operator > (const IntSet& that) const{
		if(this->n > that.n){
			return true;
		}
		else if(this->n < that.n){
			return false;
		}
		else if(this->n > 0){
			// Check if one is empty
			if(this->elements == NULL || that.elements == NULL){
				std::cout << "At least, one set is empty" << std::endl;
				return false;
			}

			for(int i=0; i<n; i++){
				if(this->elements[i] > that.elements[i]){
					return true;
				}
				else if(this->elements[i] < that.elements[i]){
					return false;
				}
			}
			return false;
		}
		else{
			return false;
		}
	}

	bool operator == (const IntSet& that) const{
		if(this->n == that.n){
			for(int i=0; i<n; i++){
				if(this->elements[i] != that.elements[i]){
					return false;
				}
			}
		}
		else{
			return false;
		}
		return true;
	}

	void print(){
		std::cout << "n = " << n << ": ";
		for(int i=0; i<n; i++){
			std::cout << elements[i] << ", ";
		}
		std::cout << std::endl;
	}

	void sorted(){
		std::sort(elements, elements+n);
	}
};


class MonIdxSet{
	friend class MonSet;

	IntSet pos;
	int eq_idx;
	int mon_idx;
	bool sys_idx;
	CT coef;

	void init(const int n, const int* elements){
		pos = IntSet(n, elements);
		eq_idx = 0;
		mon_idx = 0;
		sys_idx = 0;
		coef = 0.0;
	}

	void init(const int n, const int* elements, int eq_idx, int mon_idx, bool sys_idx){
		pos = IntSet(n, elements);
		this->eq_idx = eq_idx;
		this->mon_idx = mon_idx;
		this->sys_idx = sys_idx;
		this->coef = 0.0;
	}

	void init(const int n, const int* elements, int eq_idx, int mon_idx, bool sys_idx, const CT& coef){
		pos = IntSet(n, elements);
		this->eq_idx = eq_idx;
		this->mon_idx = mon_idx;
		this->sys_idx = sys_idx;
		this->coef = coef;
	}

	void init(const IntSet& pos, int eq_idx, int mon_idx, bool sys_idx, const CT& coef){
		this->pos = pos;
		this->eq_idx = eq_idx;
		this->mon_idx = mon_idx;
		this->sys_idx = sys_idx;
		this->coef = coef;
	}

public:
	MonIdxSet(){
		eq_idx = 0;
		mon_idx = 0;
		sys_idx = 0;
		coef = 0.0;
	};

	MonIdxSet(const int n, const int* elements){
		init(n, elements);
	}

	MonIdxSet(const int n, const int* elements, int eq_idx, int mon_idx, bool sys_idx){
		init(n, elements, eq_idx, mon_idx, sys_idx);
	}

	MonIdxSet(const int n, const int* elements, int eq_idx, int mon_idx, bool sys_idx, const CT& coef){
		init(n, elements, eq_idx, mon_idx, sys_idx, coef);
	}

	MonIdxSet(const MonIdxSet& original){
		init(original.pos, original.eq_idx, original.mon_idx, original.sys_idx, original.coef);
	}

	MonIdxSet& operator= (const MonIdxSet& original){
		init(original.pos, original.eq_idx, original.mon_idx, original.sys_idx, original.coef);
		return *this;
	}

	~MonIdxSet(){
		//std::cout << "delete MonIdxSet " << this << endl;
	}

	// Basic functions

	IntSet get_pos(){
		return pos;
	}

	int get_eq_idx(){
		return eq_idx;
	}

	int get_mon_idx(){
		return mon_idx;
	}

	bool get_sys_idx(){
		return sys_idx;
	}

	CT get_coef(){
		return coef;
	}

	// comparison operators
	bool operator < (const MonIdxSet& that) const{
		if(this->pos < that.pos){
			return true;
		}
		else if(this->pos == that.pos){
			// Compare equation index
			if(this->eq_idx < that.eq_idx){
				return true;
			}
			else if(this->eq_idx > that.eq_idx){
				return false;
			}

			// Compare system index
			if(this->sys_idx < that.sys_idx){
				return true;
			}
			else if(this->sys_idx > that.sys_idx){
				return false;
			}
			return false;
		}
		else{
			return false;
		}
	}

	bool operator > (const MonIdxSet& that) const{
		if(this->pos > that.pos){
			return true;
		}
		else if(this->pos == that.pos){
			// Compare equation index
			if(this->eq_idx > that.eq_idx){
				return true;
			}
			else if(this->eq_idx < that.eq_idx){
				return false;
			}

			// Compare system index
			if(this->sys_idx > that.sys_idx){
				return true;
			}
			else if(this->sys_idx < that.sys_idx){
				return false;
			}

			return false;
		}
		else{
			return false;
		}
	}

	bool operator == (const MonIdxSet& that) const{
		if(this->pos == that.pos){
			return true;
		}
		else{
			return false;
		}
	}

	void print(){
		pos.print();
		std::cout << "s" << sys_idx
				  << " e" << eq_idx
				  << " m" << mon_idx << std::endl;
		std::cout << coef;
		std::cout << std::endl;
	}

	void sorted(){
		pos.sorted();
	}
};

class MonSet{
	IntSet pos;
	IntSet eq_idx;
	CT* coef;

	void init(const int n, const int* elements){
		pos = IntSet(n, elements);
		coef = NULL;
	}

	void init(const int n, const int* elements, const int n_mon, const int* eq_idx){
		pos = IntSet(n, elements);
		this->eq_idx = IntSet(n_mon, eq_idx);
		coef = NULL;
	}

	void init(const MonSet& original){
		pos = original.pos;
		eq_idx = original.eq_idx;
		copy_coef(eq_idx.get_n()*2, original.coef);
	}

public:
	MonSet(){
		coef = NULL;
	};

	void copy_coef(int n, CT* coef){
		this->coef = new CT[n];
		for(int i=0; i<n; i++){
			this->coef[i] = coef[i];
		}
	}

	void copy_pos(const MonIdxSet& original){
		this->pos = original.pos;
	}

	void update_eq_idx(int n, int* eq_idx_elements){
		eq_idx.update(n,eq_idx_elements);
	}

	void update_coef(CT* tmp_coef){
		coef = tmp_coef;
	}

	MonSet(const int n, const int* elements){
		init(n, elements);
		coef = NULL;
	}

	MonSet(const int n, const int* elements, const int n_mon, const int* eq_idx){
		init(n, elements, n_mon, eq_idx);
		coef = NULL;
	}

	MonSet(const MonSet& original){
		init(original);
	}

	MonSet& operator= (const MonSet& original){
		init(original);
		return *this;
	}

	~MonSet(){
		delete[] coef;
		coef = NULL;
	}

	bool operator < (const MonSet& that) const{
		if(this->pos < that.pos){
			return true;
		}
		else{
			return false;
		}
	}

	bool operator > (const MonSet& that) const{
		if(this->pos > that.pos){
			return true;
		}
		else{
			return false;
		}
	}

	bool operator == (const MonSet& that) const{
		if(this->pos == that.pos){
			return true;
		}
		else{
			return false;
		}
	}

	void print(){
		pos.print();
		std::cout <<  "n_mon = " << eq_idx.get_n() << ": ";
		for(int i=0; i<eq_idx.get_n(); i++){
			std::cout << " e" << eq_idx.get_element(i) << std::endl;
			std::cout << coef[2*i] << coef[2*i+1];
		}
		std::cout << std::endl;
	}

	void sorted(){
		pos.sorted();
	}
};
