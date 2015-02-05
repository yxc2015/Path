QD_ROOT=/usr/local/qd-2.3.13
QD_LIB=/usr/local/lib
GQD_HOME=/usr/local/gqd_1_2
SDK_HOME=/usr/local/cuda/sdk

all: path

gqd_qd_util.o: Complex/gqd_qd_util.h Complex/gqd_qd_util.cpp
	@-echo ">>> compiling utilities ..."
	g++ -O2 -I/usr/local/cuda/include -I$(GQD_HOME)/inc \
            -I$(QD_ROOT)/include -c Complex/gqd_qd_util.cpp
            
# D

workspace_host_d.o: Path_CPU/workspace_host.h Path_CPU/workspace_host.cpp
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/workspace_host.cpp -o workspace_host_d.o

eval_host_d.o: Path_CPU/eval_host.h Path_CPU/eval_host.cpp poly_d.o workspace_host_d.o DefineTypesDD/DefineType.h
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/eval_host.cpp -o eval_host_d.o

path_host_d.o: Path_CPU/path_host.h Path_CPU/path_host.cpp predictor_host_d.o newton_host_d.o
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/path_host.cpp -o path_host_d.o

newton_host_d.o: Path_CPU/newton_host.h Path_CPU/newton_host.cpp eval_host_d.o mgs_host_d.o 
	g++ -O2 -I./DefineTypesD  -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/newton_host.cpp -o newton_host_d.o

predictor_host_d.o: Path_CPU/predictor_host.h Path_CPU/predictor_host.cpp workspace_host_d.o
	g++ -O2 -I./DefineTypesD  -I$(QD_ROOT)/include \
            -c Path_CPU/predictor_host.cpp -o predictor_host_d.o

mgs_host_d.o: Path_CPU/mgs_host.h Path_CPU/mgs_host.cpp 
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/mgs_host.cpp -o mgs_host_d.o

path_gpu_d.o: Path_GPU/*.cu Path_GPU/*.h DefineTypesDD/DefineType.h
	@-echo ">>> compiling kernels ..."
	nvcc -O2  --define-macro path_precision=cd -I./DefineTypesD \
             -I./Path_CPU -I./Complex \
             -I$(GQD_HOME)/inc -I$(SDK_HOME)/C/common/inc \
             -I/usr/local/cuda/include -o path_gpu_d.o \
             -c Path_GPU/path_gpu.cu --ptxas-options=-v

utilities_d.o: Path_CPU/utilities.h Path_CPU/utilities.cpp
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/utilities.cpp -o utilities_d.o 

families_d.o: Path_CPU/families.h Path_CPU/families.cpp
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/families.cpp -o families_d.o

poly_d.o: Path_CPU/poly.h Path_CPU/poly.cpp utilities_d.o DefineTypesDD/DefineType.h
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/poly.cpp -o poly_d.o
	         
path_d.o: Path_CPU/path.cpp Path_CPU/parameter.h families_d.o path_host_d.o path_gpu_d.o newton_host_d.o
	g++ -O2  -I./DefineTypesD -I./Path_GPU -I./Path_CPU \
            -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/path.cpp -o path_d.o
	         
path_d: families_d.o path_d.o poly_d.o utilities_d.o path_gpu_d.o gqd_qd_util.o path_host_d.o eval_host_d.o
	g++ -O2 -I$(GQD_HOME)/inc -I$(QD_ROOT)/include \
		path_d.o families_d.o poly_d.o utilities_d.o path_gpu_d.o gqd_qd_util.o\
		predictor_host_d.o mgs_host_d.o newton_host_d.o path_host_d.o eval_host_d.o\
		workspace_host_d.o\
		$(QD_LIB)/libqd.a -o path_d \
		-lcutil_x86_64 -lcudart \
		-L/usr/local/cuda/lib64 -L$(SDK_HOME)/C/lib

            
# DD

workspace_host_dd.o: Path_CPU/workspace_host.h Path_CPU/workspace_host.cpp
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/workspace_host.cpp -o workspace_host_dd.o

eval_host_dd.o: Path_CPU/eval_host.h Path_CPU/eval_host.cpp poly_dd.o workspace_host_dd.o DefineTypesDD/DefineType.h
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/eval_host.cpp -o eval_host_dd.o

path_host_dd.o: Path_CPU/path_host.h Path_CPU/path_host.cpp predictor_host_dd.o newton_host_dd.o
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/path_host.cpp -o path_host_dd.o

newton_host_dd.o: Path_CPU/newton_host.h Path_CPU/newton_host.cpp eval_host_dd.o mgs_host_dd.o 
	g++ -O2 -I./DefineTypesDD  -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/newton_host.cpp -o newton_host_dd.o

predictor_host_dd.o: Path_CPU/predictor_host.h Path_CPU/predictor_host.cpp workspace_host_dd.o
	g++ -O2 -I./DefineTypesDD  -I$(QD_ROOT)/include \
            -c Path_CPU/predictor_host.cpp -o predictor_host_dd.o

mgs_host_dd.o: Path_CPU/mgs_host.h Path_CPU/mgs_host.cpp 
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/mgs_host.cpp -o mgs_host_dd.o

path_gpu_dd.o: Path_GPU/*.cu Path_GPU/*.h DefineTypesDD/DefineType.h
	@-echo ">>> compiling kernels ..."
	nvcc -O2  --define-macro path_precision=cdd -I./DefineTypesDD \
             -I./Path_CPU -I./Complex \
             -I$(GQD_HOME)/inc -I$(SDK_HOME)/C/common/inc \
             -I/usr/local/cuda/include -o path_gpu_dd.o \
             -c Path_GPU/path_gpu.cu --ptxas-options=-v

utilities_dd.o: Path_CPU/utilities.h Path_CPU/utilities.cpp
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/utilities.cpp -o utilities_dd.o 

families_dd.o: Path_CPU/families.h Path_CPU/families.cpp
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/families.cpp -o families_dd.o

poly_dd.o: Path_CPU/poly.h Path_CPU/poly.cpp utilities_dd.o DefineTypesDD/DefineType.h
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/poly.cpp -o poly_dd.o
	         
path_dd.o: Path_CPU/path.cpp Path_CPU/parameter.h families_dd.o path_host_dd.o path_gpu_dd.o newton_host_dd.o
	g++ -O2  -I./DefineTypesDD -I./Path_GPU -I./Path_CPU \
            -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/path.cpp -o path_dd.o
	         
path_dd: families_dd.o path_dd.o poly_dd.o utilities_dd.o path_gpu_dd.o gqd_qd_util.o path_host_dd.o eval_host_dd.o
	g++ -O2 -I$(GQD_HOME)/inc -I$(QD_ROOT)/include \
		path_dd.o families_dd.o poly_dd.o utilities_dd.o path_gpu_dd.o gqd_qd_util.o\
		predictor_host_dd.o mgs_host_dd.o newton_host_dd.o path_host_dd.o eval_host_dd.o\
		workspace_host_dd.o\
		$(QD_LIB)/libqd.a -o path_dd \
		-lcutil_x86_64 -lcudart \
		-L/usr/local/cuda/lib64 -L$(SDK_HOME)/C/lib

# QD

workspace_host_qd.o: Path_CPU/workspace_host.h Path_CPU/workspace_host.cpp
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/workspace_host.cpp -o workspace_host_qd.o

eval_host_qd.o: Path_CPU/eval_host.h Path_CPU/eval_host.cpp poly_qd.o workspace_host_qd.o DefineTypesDD/DefineType.h
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/eval_host.cpp -o eval_host_qd.o

path_host_qd.o: Path_CPU/path_host.h Path_CPU/path_host.cpp predictor_host_qd.o newton_host_qd.o
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/path_host.cpp -o path_host_qd.o

newton_host_qd.o: Path_CPU/newton_host.h Path_CPU/newton_host.cpp eval_host_qd.o mgs_host_qd.o 
	g++ -O2 -I./DefineTypesQD  -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/newton_host.cpp -o newton_host_qd.o

predictor_host_qd.o: Path_CPU/predictor_host.h Path_CPU/predictor_host.cpp workspace_host_qd.o
	g++ -O2 -I./DefineTypesQD  -I$(QD_ROOT)/include \
            -c Path_CPU/predictor_host.cpp -o predictor_host_qd.o

mgs_host_qd.o: Path_CPU/mgs_host.h Path_CPU/mgs_host.cpp 
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/mgs_host.cpp -o mgs_host_qd.o

path_gpu_qd.o: Path_GPU/*.cu Path_GPU/*.h DefineTypesDD/DefineType.h
	@-echo ">>> compiling kernels ..."
	nvcc -O2  --define-macro path_precision=cqd -I./DefineTypesQD \
             -I./Path_CPU -I./Complex \
             -I$(GQD_HOME)/inc -I$(SDK_HOME)/C/common/inc \
             -I/usr/local/cuda/include -o path_gpu_qd.o \
             -c Path_GPU/path_gpu.cu --ptxas-options=-v

utilities_qd.o: Path_CPU/utilities.h Path_CPU/utilities.cpp
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/utilities.cpp -o utilities_qd.o 

families_qd.o: Path_CPU/families.h Path_CPU/families.cpp
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/families.cpp -o families_qd.o

poly_qd.o: Path_CPU/poly.h Path_CPU/poly.cpp utilities_qd.o DefineTypesDD/DefineType.h
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/poly.cpp -o poly_qd.o
	         
path_qd.o: Path_CPU/path.cpp Path_CPU/parameter.h families_qd.o path_host_qd.o path_gpu_qd.o newton_host_qd.o
	g++ -O2  -I./DefineTypesQD -I./Path_GPU -I./Path_CPU \
            -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c Path_CPU/path.cpp -o path_qd.o
	         
path_qd: families_qd.o path_qd.o poly_qd.o utilities_qd.o path_gpu_qd.o gqd_qd_util.o path_host_qd.o eval_host_qd.o
	g++ -O2 -I$(GQD_HOME)/inc -I$(QD_ROOT)/include \
		path_qd.o families_qd.o poly_qd.o utilities_qd.o path_gpu_qd.o gqd_qd_util.o\
		predictor_host_qd.o mgs_host_qd.o newton_host_qd.o path_host_qd.o eval_host_qd.o\
		workspace_host_qd.o\
		$(QD_LIB)/libqd.a -o path_qd \
		-lcutil_x86_64 -lcudart \
		-L/usr/local/cuda/lib64 -L$(SDK_HOME)/C/lib
clean:
	rm -rf *.o path_d path_dd path_qd
