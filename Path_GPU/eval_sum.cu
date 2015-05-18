#include "eval_sum_seq.cu"
#include "eval_sum_tree.cu"

void eval_sum(GPUWorkspace& workspace, const GPUInst& inst, int n_sys){
	int sum_method = 0;
	if(sum_method == 0){
		eval_sum_seq(workspace, inst, n_sys);
	}
	else{
		eval_sum_tree(workspace, inst, n_sys);
	}
}
