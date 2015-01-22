__global__ void get_max_array_double(complex<gdd_real>* sol, int dim, int dimLog2, double* max_delta_x ) {
	__shared__ double delta_x[max_array_size];

	int j = threadIdx.x;
	// max for the norm
	delta_x[j] = sol[j].real.x*sol[j].real.x + sol[j].imag.x*sol[j].imag.x;

	dimLog2 -= 1;
	int half_size = 1 << (dimLog2);// sum for the norm

	if(half_size > 16) {
		__syncthreads();
	}
	if(j + half_size < dim) {
		if(delta_x[j] < delta_x[j+half_size]) {
			delta_x[j] = delta_x[j+half_size];
		}
	}
	for(int k=0; k < dimLog2; k++) {
		if(half_size > 16) {
			__syncthreads();
		}
		half_size /= 2;
		if(j < half_size) {
			if(delta_x[j] < delta_x[j+half_size]) {
				delta_x[j] = delta_x[j+half_size];
			}
		}
	}

	if(j == 0) {
		*max_delta_x = delta_x[0];
	}
}

__global__ void small_backsubstitution_with_delta(complex<gdd_real>* R, complex<gdd_real>* sol0, int dim, int dimLog2, double* max_delta_x)
{
	int j = threadIdx.x;
	__shared__ complex<gdd_real> sol[shmemsize/2];
	__shared__ complex<gdd_real> Rcl[shmemsize/2];
	__shared__ double delta_x[shmemsize/2];
	int ind;
	complex<gdd_real> update;

	update = R[j];
	for(int k=dim-1; k>=0; k--)  // compute k-th component of solution
	{
		if(j < k+1)
		{
			ind = (dim - k)*(dim + 3 + k)/2 + j;
			Rcl[j] = R[ind];
		}
		if(j == k) sol[j] = update/Rcl[j]; // all other threads wait
		__syncthreads();
		if(j < k) update = update - sol[k]*Rcl[j];// update
	}
	sol0[j] = sol[j];

	// max for the norm
	delta_x[j] = sol[j].real.x*sol[j].real.x + sol[j].imag.x*sol[j].imag.x;
	__syncthreads();

	dimLog2 -= 1;
	int half_size = 1 << (dimLog2);// sum for the norm
	if(j + half_size < dim) {
		if(delta_x[j] < delta_x[j+half_size]) {
			delta_x[j] = delta_x[j+half_size];
		}
	}
	for(int k=0; k < dimLog2; k++) {
		if(half_size > 16) {
			__syncthreads();
		}
		half_size /= 2;
		if(j < half_size) {
			if(delta_x[j] < delta_x[j+half_size]) {
				delta_x[j] = delta_x[j+half_size];
			}
		}
	}
	if(j == 0) {
		*max_delta_x = delta_x[0];
	}
}


// Mon evalutaion and differentiation on GPU
__global__ void eval_mon_level0_kernel(GT* workspace_mon, GT* x, GT*workspace_coef,
                           int* mon_pos_start, unsigned short* mon_pos, int n_mon) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int sys_idx = blockIdx.z;
	//GT* x_d_tmp = x_d + sys_idx*dim;
	//GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

	//int tidx = threadIdx.x;
	if(idx < n_mon) {
		int tmp_start = mon_pos_start[idx];
		GT* deri = workspace_mon + tmp_start;
		unsigned short* pos = mon_pos + tmp_start;

		GT tmp = workspace_coef[idx];
		deri[1] = tmp;
		deri[0] = x[pos[1]]*tmp;
	}
}

// Mon evalutaion and differentiation on GPU
__global__ void eval_mon_global_kernel(GT* workspace_mon, GT* x, GT* workspace_coef,
int* mon_pos_start, unsigned short* mon_pos, int n_mon) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//int sys_idx = blockIdx.z;
	//GT* x_d_tmp = x_d + sys_idx*dim;
	//GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

	//int tidx = threadIdx.x;
	if(idx < n_mon) {
		int tmp_start = mon_pos_start[idx];
		GT* deri = workspace_mon + tmp_start;
		unsigned short* pos = mon_pos + tmp_start;

		int n_var = pos[0];

		GT tmp = x[pos[1]];

		GT* deri_tmp = deri + 1;
		deri_tmp[1] = tmp;

		for(int i=2; i<n_var; i++) {
			tmp *= x[pos[i]];
			deri_tmp[i] = tmp;
		}

		tmp = workspace_coef[idx];

		for(int i=n_var; i>1; i--) {
			deri[i] *= tmp;
			tmp *= x[pos[i]];
		}
		deri[1] = tmp;
		deri[0] = x[pos[1]]*tmp;
	}
}

// Mon evalutation and differentiation on GPU, level 0 x[i] + 1 as its derivative
__global__ void mon_block_unroll1(GT* workspace_d, GT* x_d,  GT* workspace_coef,\
		                          int* mon_pos_start,\
		                          unsigned short* pos_d, int n_mon){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < n_mon){
        //int sys_idx = blockIdx.z;
        //GT* x_d_tmp = x_d + sys_idx*dim;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;
		int tmp_start = mon_pos_start[idx];
		GT* deri = workspace_d + tmp_start;
		unsigned short* pos = pos_d + tmp_start;

		GT tmp = workspace_coef[idx];
		deri[1] = tmp;
		deri[0] = x_d[pos[1]]*tmp;
    }
}

// Mon evalutation and differentiation on GPU, level 1 x0*x1, x1, x0
__global__ void mon_block_unroll2(GT* workspace_d, GT* x_d, GT* workspace_coef,
        						  int* mon_pos_start,\
		                          unsigned short* pos_d, int n_mon){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < n_mon){
        //int sys_idx = blockIdx.z;
        //GT* x_d_tmp = x_d + sys_idx*dim;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

        //int xidx = idx*3;
		int xidx = mon_pos_start[idx];

        // Load to Shared Memory
        GT x0, x1, tmp;

        x1  = x_d[pos_d[xidx+2]];
        tmp = workspace_coef[idx];
        x0  = x_d[pos_d[xidx+1]]*tmp;

        workspace_d[xidx+2] = x0;
        workspace_d[xidx]   = x0*x1;
        workspace_d[xidx+1] = x1*tmp;
    }
}

// Sum block, mulitithread gpu sum unroll, for test
template <unsigned int n_th>
__global__ void mon_block_unroll(GT* workspace_d, GT* x_d, GT* workspace_coef,
								 int* mon_pos_start,\
								 unsigned short* pos_d, int n_mon){
        //GT* x_d, unsigned short* pos_d, GT* workspace_d, int n_mon,
        //                        int dim, int workspace_size_int){
    __shared__ GT x_sh[shmemsize];
    int BS = blockDim.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*BS;
    //int idx = bidx + threadIdx.x;
    int tidx = threadIdx.x;
    //int tidx2 = tidx + BS;

    int midx = tidx/n_th; // monomial index
    int midx_global =  midx + bidx/n_th;

    if(midx_global < n_mon){
        //int sys_idx = blockIdx.z;
        //GT* x_d_tmp = x_d + sys_idx*dim;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

        int pidx = tidx - midx*n_th; // thread index in monomial
        int pidx2 = pidx + n_th;
        //int xidx0 = midx_global*(n_th*2+1);
		int xidx0 = mon_pos_start[midx_global];
        int xidx1 = xidx0+ pidx+1; // pos index 1
        int xidx2 = xidx1 + n_th;  // pos index 2

        //int* pos = pos_d;// + BS*2*blockIdx.x;
        int n = pos_d[xidx0];

        GT* x_level = x_sh + midx*n_th*2;

        // Load to Shared Memory
        GT x0, x1;
        x0 = x_d[pos_d[xidx1]];
        if(pidx2 < n){
            x1 = x_d[pos_d[xidx2]];
            x_level[pidx] = x0*x1;
        }
        else{
            x_level[pidx] = x0;
        }

        if(n_th > 32){
            __syncthreads();
        }

        // Up
        GT* x_last_level;

        if(n_th > 256){
            x_last_level = x_level; x_level += 512;
            if(pidx < 256){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+256];}
            __syncthreads();
        }

        if(n_th > 128){
            x_last_level = x_level; x_level += 256;
            if(pidx < 128){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+128];}
            __syncthreads();
        }

        if(n_th > 64){
            x_last_level = x_level; x_level += 128;
            if(pidx < 64){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+64];}
            __syncthreads();
        }

        if(n_th > 32){
            x_last_level = x_level; x_level += 64;
            if(pidx < 32){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+32];}
        }

        if(n_th > 16){
            x_last_level = x_level; x_level += 32;
            if(pidx < 16){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+16];}
        }

        if(n_th > 8){
            x_last_level = x_level; x_level += 16;
            if(pidx <  8){x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 8];}
        }

        if(n_th > 4){
            x_last_level = x_level; x_level +=  8;
            if(pidx <  4){  x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 4];}
        }

        if(n_th > 2){
            x_last_level = x_level; x_level +=  4;
            if(pidx <  2){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 2];}
        }

        if(n_th > 1){
            x_last_level = x_level; x_level +=  2;
            if(pidx <  1){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 1];}
        }

        if(pidx < 3){
        	x_last_level[pidx] *= workspace_coef[midx];
        }

        // Common Factor to be multiplied
        if(pidx == 0){
            workspace_d[xidx0] = x_level[0];
        }

        // Down
        if(n_th > 2){
            x_level = x_last_level;
            x_last_level -= 4;
            if(pidx <  4){
                int new_pidx = (pidx&1) ^ 1; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
        }

        if(n_th > 4){
            x_level = x_last_level;
            x_last_level -= 8;
            if(pidx <  8){
                int new_pidx = (pidx&3) ^ 2; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
        }


        if(n_th > 8){
            x_level = x_last_level;
            x_last_level -= 16;
            if(pidx <  16){
                int new_pidx = (pidx & 7) ^ 4; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
        }

        if(n_th > 16){
            x_level = x_last_level;
            x_last_level -= 32;
            if(pidx < 32){
                int new_pidx = (pidx & 15) ^ 8; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 32){
            x_level = x_last_level;
            x_last_level -= 64;
            if(pidx < 64){
                int new_pidx = (pidx & 31) ^ 16; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 64){
            x_level = x_last_level;
            x_last_level -= 128;
            if(pidx < 128){
                int new_pidx = (pidx & 63) ^ 32; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 128){
            x_level = x_last_level;
            x_last_level -= 256;
            if(pidx < 256){
                int new_pidx = (pidx & 127) ^ 64; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 256){
            x_level = x_last_level;
            x_last_level -= 512;
            if(pidx < 512){
                int new_pidx = (pidx & 255) ^ 128; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        // Copy Derivative
        //bidx *= 2;
        //tidx += 1; // for test, keep tidx in memory
        int new_tidx = pidx ^ (n_th/2); // XOR to swith the term in last level
        GT x_upper = x_last_level[new_tidx];
        if(pidx2 < n){
             workspace_d[xidx1] = x1*x_upper;
             workspace_d[xidx2] = x0*x_upper;
        }
        else{
             workspace_d[xidx1] = x_upper;
        }
    }
}

// Sum block, mulitithread gpu sum unroll, for test
template <unsigned int n_th>
__global__ void mon_block_unroll_n(GT* workspace_d, GT* x_d, GT* workspace_coef,
								 int* mon_pos_start,\
								 unsigned short* pos_d, int n_mon){
        //                        int dim, int workspace_size_int){
    __shared__ GT x_sh[shmemsize];
    int BS = blockDim.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*BS;
    //int idx = bidx + threadIdx.x;
    int tidx = threadIdx.x;
    //int tidx2 = tidx + BS;

    int midx = tidx/n_th; // monomial index
    int midx_global =  midx + bidx/n_th;

    if(midx_global < n_mon){
        //int sys_idx = blockIdx.z;
        //GT* x_d_tmp = x_d + sys_idx*dim;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

        int pidx = tidx - midx*n_th; // thread index in monomial
        //int xidx0 = midx_global*(n_th*2+1);
		int xidx0 = mon_pos_start[midx_global];

        //int* pos = pos_d;// + BS*2*blockIdx.x;
        int n = pos_d[xidx0];

        GT* x_level = x_sh + midx*n_th*2;

        // Load to Shared Memory
        unsigned short* pos_tmp = pos_d+xidx0+1;
        GT* workspace_tmp = workspace_d+xidx0+1;

        int pos_idx = pidx;
        GT tmp = x_d[pos_tmp[pos_idx]];
        workspace_tmp[pos_idx] = GT(1,0);
    	pos_idx += n_th;
        while(pos_idx < n){
            workspace_tmp[pos_idx] = tmp;
        	tmp *= x_d[pos_tmp[pos_idx]];
        	pos_idx += n_th;
        }
        x_level[pidx] = tmp;

        // Up
        GT* x_last_level;

        if(n_th > 256){
            __syncthreads();
            x_last_level = x_level; x_level += 512;
            if(pidx < 256){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+256];}
        }

        if(n_th > 128){
            __syncthreads();
            x_last_level = x_level; x_level += 256;
            if(pidx < 128){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+128];}
        }

        if(n_th > 64){
            __syncthreads();
            x_last_level = x_level; x_level += 128;
            if(pidx < 64){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+64];}
        }

        if(n_th > 32){
            __syncthreads();
            x_last_level = x_level; x_level += 64;
            if(pidx < 32){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+32];}
        }

        if(n_th > 16){
            x_last_level = x_level; x_level += 32;
            if(pidx < 16){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+16];}
        }

        if(n_th > 8){
            x_last_level = x_level; x_level += 16;
            if(pidx <  8){x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 8];}
        }

        if(n_th > 4){
            x_last_level = x_level; x_level +=  8;
            if(pidx <  4){  x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 4];}
        }

        if(n_th > 2){
            x_last_level = x_level; x_level +=  4;
            if(pidx <  2){ x_level[pidx] = x_last_level[pidx] * x_last_level[pidx+ 2];}
        }

        if(n_th > 1){
            x_last_level = x_level; x_level +=  2;
            if(pidx == 0){ x_level[0] = x_last_level[0] * x_last_level[1];}
        }

        if(pidx < 3){
        	x_last_level[pidx] *= workspace_coef[midx_global];
        }

        // Common Factor to be multiplied
        if(pidx == 0){
            workspace_d[xidx0] = x_level[0];
        }

        // Down
        if(n_th > 2){
            x_level = x_last_level;
            x_last_level -= 4;
            if(pidx <  4){
                int new_pidx = (pidx&1) ^ 1; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
        }

        if(n_th > 4){
            x_level = x_last_level;
            x_last_level -= 8;
            if(pidx <  8){
                int new_pidx = (pidx&3) ^ 2; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
        }


        if(n_th > 8){
            x_level = x_last_level;
            x_last_level -= 16;
            if(pidx <  16){
                int new_pidx = (pidx & 7) ^ 4; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
        }

        if(n_th > 16){
            x_level = x_last_level;
            x_last_level -= 32;
            if(pidx < 32){
                int new_pidx = (pidx & 15) ^ 8; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 32){
            x_level = x_last_level;
            x_last_level -= 64;
            if(pidx < 64){
                int new_pidx = (pidx & 31) ^ 16; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 64){
            x_level = x_last_level;
            x_last_level -= 128;
            if(pidx < 128){
                int new_pidx = (pidx & 63) ^ 32; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 128){
            x_level = x_last_level;
            x_last_level -= 256;
            if(pidx < 256){
                int new_pidx = (pidx & 127) ^ 64; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        if(n_th > 256){
            x_level = x_last_level;
            x_last_level -= 512;
            if(pidx < 512){
                int new_pidx = (pidx & 255) ^ 128; // XOR to swith the term in last level
                x_last_level[pidx]*=x_level[new_pidx];
            }
            __syncthreads();
        }

        // Copy Derivative
        //bidx *= 2;
        //tidx += 1; // for test, keep tidx in memory
        int new_tidx = pidx ^ (n_th/2); // XOR to swith the term in last level
        tmp = x_last_level[new_tidx];

        pos_idx = pidx + (n-1)/n_th*n_th;
        //int tmp_idx = pos_idx;

        if(pos_idx < n){
        	workspace_tmp[pos_idx] *= tmp;
			tmp *= x_d[pos_tmp[pos_idx]];
        }
        pos_idx -= n_th;
        //GT tmp = x_d[pos_tmp[pos_idx]];
        //workspace_tmp[pos_idx] = tmp;

        while(pos_idx >= n_th){
        	workspace_tmp[pos_idx] *= tmp;
			tmp *= x_d[pos_tmp[pos_idx]];
        	pos_idx -= n_th;
        }
    	workspace_tmp[pos_idx] *= tmp;
    }
}

// Sum block, mulitithread gpu sum unroll, for test
__global__ void mon_block_unroll_4(GT* workspace_d, GT* x_d, GT* workspace_coef,
								 int* mon_pos_start,\
								 unsigned short* pos_d, int n_mon){
        //GT* x_d, unsigned short* pos_d, GT* workspace_d, int n_mon,
        //                        int dim, int workspace_size_int){
    __shared__ GT x_sh[shmemsize];
    int BS = blockDim.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*BS;
    //int idx = bidx + threadIdx.x;
    int tidx = threadIdx.x;
    //int tidx2 = tidx + BS;

    int midx = tidx/2; // monomial index
    int midx_global =  midx + bidx/2;

    if(midx_global < n_mon){
        //int sys_idx = blockIdx.z;
        //GT* x_d_tmp = x_d + sys_idx*dim;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

        int pidx = tidx - midx*2; // thread index in monomial
        int pidx2 = pidx + 2;
        //int xidx0 = midx_global*(n_th*2+1);
		int xidx0 = mon_pos_start[midx_global];
        int xidx1 = xidx0+ pidx+1; // pos index 1
        int xidx2 = xidx1 + 2;  // pos index 2

        //int* pos = pos_d;// + BS*2*blockIdx.x;
        int n = pos_d[xidx0];

        GT* x_level = x_sh + midx*3;

        // Load to Shared Memory
        GT x0, x1;
        x0 = x_d[pos_d[xidx1]];
        if(pidx2 < n){
            x1 = x_d[pos_d[xidx2]];
            x_level[pidx] = x0*x1;
        }
        else{
            x_level[pidx] = x0;
        }

		if(pidx == 0){
			x_level[2] = x_level[0] * x_level[1];
		}

		GT coef_tmp = workspace_coef[midx_global];

		x_level[pidx] *= coef_tmp;

		if(pidx == 0){
			workspace_d[xidx0] = x_level[2]*coef_tmp;
		}

        // Copy Derivative
        int new_tidx = pidx ^ 1; // XOR to swith the term in last level
        GT x_upper = x_level[new_tidx];
        if(pidx2 < n){
             workspace_d[xidx1] = x1*x_upper;
             workspace_d[xidx2] = x0*x_upper;
        }
        else{
             workspace_d[xidx1] = x_upper;
        }
    }
}


// Sum block, mulitithread gpu sum unroll, for test
__global__ void mon_block_unroll_4n(GT* workspace_d, GT* x_d, GT* workspace_coef,
								 int* mon_pos_start,\
								 unsigned short* pos_d, int n_mon){
        //GT* x_d, unsigned short* pos_d, GT* workspace_d, int n_mon,
        //                        int dim, int workspace_size_int){
    __shared__ GT x_sh[shmemsize];
    int BS = blockDim.x;
    int bidx = (gridDim.x*blockIdx.y+blockIdx.x)*BS;
    //int idx = bidx + threadIdx.x;
    int tidx = threadIdx.x;
    //int tidx2 = tidx + BS;

    int midx = tidx/2; // monomial index
    int midx_global =  midx + bidx/2;

    if(midx_global < n_mon){
        //int sys_idx = blockIdx.z;
        //GT* x_d_tmp = x_d + sys_idx*dim;
        //GT* workspace_d_tmp = workspace_d + sys_idx*workspace_size_int;

        int pidx = tidx - midx*2; // thread index in monomial
        //int pidx2 = pidx + 2;
        //int xidx0 = midx_global*(n_th*2+1);
		int xidx0 = mon_pos_start[midx_global];
        //int xidx1 = xidx0+ pidx+1; // pos index 1
        //int xidx2 = xidx1 + 2;  // pos index 2

        //int* pos = pos_d;// + BS*2*blockIdx.x;
        int n = pos_d[xidx0];

        GT* x_level = x_sh + midx*3;

        // Load to Shared Memory
        unsigned short* pos_tmp = pos_d+xidx0+1;
        GT* workspace_tmp = workspace_d+xidx0+1;

        int pos_idx = pidx;
        GT tmp = x_d[pos_tmp[pos_idx]];
        workspace_tmp[pos_idx] = GT(1,0);
    	pos_idx += 2;
        while(pos_idx < n){
            workspace_tmp[pos_idx] = tmp;
        	tmp *= x_d[pos_tmp[pos_idx]];
        	pos_idx += 2;
        }
        x_level[pidx] = tmp;

		if(pidx == 0){
			x_level[2] = x_level[0] * x_level[1];
		}

		GT coef_tmp = workspace_coef[midx_global];

		x_level[pidx] *= coef_tmp;

		if(pidx == 0){
			workspace_d[xidx0] = x_level[2]*coef_tmp;
		}

        // Copy Derivative
        int new_tidx = pidx ^ 1; // XOR to swith the term in last level
        tmp = x_level[new_tidx];


        pos_idx = pidx + (n-1)/2*2;
        //int tmp_idx = pos_idx;

        if(pos_idx < n){
        	workspace_tmp[pos_idx] *= tmp;
			tmp *= x_d[pos_tmp[pos_idx]];
        }
        pos_idx -= 2;
        //GT tmp = x_d[pos_tmp[pos_idx]];
        //workspace_tmp[pos_idx] = tmp;

        while(pos_idx >= 2){
        	workspace_tmp[pos_idx] *= tmp;
			tmp *= x_d[pos_tmp[pos_idx]];
        	pos_idx -= 2;
        }
    	workspace_tmp[pos_idx] *= tmp;
    }
}

void eval_mon(GPUWorkspace& workspace, const GPUInst& inst, int n_sys){
	int eval_method = 2;
	if(eval_method == 0){
		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
				inst.mon_pos, inst.n_mon_level[0]);

		eval_mon_global_kernel<<<inst.mon_global_grid, inst.mon_global_BS>>>(
				workspace.mon, workspace.x, workspace.coef + inst.n_mon_level[0],
				inst.mon_pos_start + inst.n_mon_level[0], inst.mon_pos,
				inst.n_mon_global);
	}
	else if(eval_method == 1){
		std::cout << "inst.level = " << inst.level << std::endl;
		int max_level = 8;

		int* pos_start_tmp = inst.mon_pos_start;
		GT* workspace_coef_tmp = workspace.coef;

		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
				inst.mon_pos, inst.n_mon_level[0]);

		pos_start_tmp += inst.n_mon_level[0];
		workspace_coef_tmp += inst.n_mon_level[0];

		if(inst.level > 1){
			int n_mon_tmp = inst.n_mon_level[1];
			mon_block_unroll2<<<inst.mon_level_grid[1], inst.mon_global_BS>>>(
					workspace.mon, workspace.x, workspace_coef_tmp,
					pos_start_tmp, inst.mon_pos,
					n_mon_tmp);
			pos_start_tmp += n_mon_tmp;
			workspace_coef_tmp += n_mon_tmp;
		}

	    if(inst.level > 2){
			int n_mon_tmp = inst.n_mon_level[2];
			mon_block_unroll_4<<<inst.mon_level_grid[2], inst.mon_level_BS>>>(
					workspace.mon, workspace.x, workspace_coef_tmp,
					pos_start_tmp, inst.mon_pos,
					n_mon_tmp);
			pos_start_tmp += n_mon_tmp;
			workspace_coef_tmp += n_mon_tmp;
		}

	    int last_level = min(inst.level, max_level+1);
	    for(int i=3; i<last_level; i++){
			if(i==3){
				mon_block_unroll<4><<<inst.mon_level_grid[3], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==4){
				mon_block_unroll<8><<<inst.mon_level_grid[4], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==5){
				mon_block_unroll<16><<<inst.mon_level_grid[5], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==6){
				mon_block_unroll<32><<<inst.mon_level_grid[6], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==7){
				mon_block_unroll<64><<<inst.mon_level_grid[7], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==8){
				mon_block_unroll<128><<<inst.mon_level_grid[8], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			/*if(i==9){
				mon_block_unroll<256><<<inst.mon_level_grid[9], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}*/
			pos_start_tmp += inst.n_mon_level[i];
			workspace_coef_tmp += inst.n_mon_level[i];
	    }

	    // To be tested and improved by tree structure by sequential bottom level
	    if(inst.n_mon_level_rest[last_level-1] > 0){
			eval_mon_global_kernel<<<inst.mon_level_grid_rest[last_level-1], inst.mon_global_BS>>>(
					workspace.mon, workspace.x, workspace_coef_tmp,
					pos_start_tmp, inst.mon_pos, inst.n_mon_level_rest[last_level-1]);
	    }
    }
	else{
		std::cout << "Eval NEW" << std::endl;
		std::cout << "inst.level = " << inst.level << std::endl;
		int max_level = 3;

		int* pos_start_tmp = inst.mon_pos_start;
		GT* workspace_coef_tmp = workspace.coef;

	    int last_level = min(inst.level, max_level+1);
	    for(int i=0; i<last_level; i++){
	    	if(i==0){
	    		eval_mon_level0_kernel<<<inst.mon_level_grid[0], inst.mon_level0_BS>>>(
	    				workspace.mon, workspace.x, workspace.coef, inst.mon_pos_start,
	    				inst.mon_pos, inst.n_mon_level[0]);
	    	}
	    	if(i==1){
				mon_block_unroll2<<<inst.mon_level_grid[1], inst.mon_global_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[1]);

	    	}
	    	if(i==2){
				mon_block_unroll_4<<<inst.mon_level_grid[2], inst.mon_level_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
	    	}
			if(i==3){
				mon_block_unroll<4><<<inst.mon_level_grid[3], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==4){
				mon_block_unroll<8><<<inst.mon_level_grid[4], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==5){
				mon_block_unroll<16><<<inst.mon_level_grid[5], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==6){
				mon_block_unroll<32><<<inst.mon_level_grid[6], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==7){
				mon_block_unroll<64><<<inst.mon_level_grid[7], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			if(i==8){
				mon_block_unroll<128><<<inst.mon_level_grid[8], inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, inst.n_mon_level[i]);
			}
			pos_start_tmp += inst.n_mon_level[i];
			workspace_coef_tmp += inst.n_mon_level[i];
	    }

	    if(inst.level > max_level+1){
			int n_mon_tmp = inst.n_mon_level_rest[max_level];
	    	if(max_level == 1){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_global_BS, n_sys, 1);
				eval_mon_global_kernel<<<mon_level_rest_grid, inst.mon_global_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, n_mon_tmp);

	    	}
	    	else if(max_level == 2){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 2);
				mon_block_unroll_4n<<<mon_level_rest_grid, inst.mon_level_BS>>>(
						workspace.mon, workspace.x, workspace_coef_tmp,
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 3){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 4);
				mon_block_unroll_n<4><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 4){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 8);
				mon_block_unroll_n<8><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 5){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 16);
				mon_block_unroll_n<16><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 6){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 32);
				mon_block_unroll_n<32><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 7){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 64);
				mon_block_unroll_n<64><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    	else if(max_level == 8){
				dim3 mon_level_rest_grid = get_grid(n_mon_tmp, inst.mon_level_BS, n_sys, 128);
				mon_block_unroll_n<128><<<mon_level_rest_grid, inst.mon_level_BS>>>(workspace.mon, workspace.x, workspace_coef_tmp,\
						pos_start_tmp, inst.mon_pos, n_mon_tmp);
	    	}
	    }
	}
}
