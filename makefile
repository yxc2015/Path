QD_ROOT=/usr/local/qd-2.3.13
QD_LIB=/usr/local/lib
GQD_HOME=/usr/local/gqd_1_2
SDK_HOME=/usr/local/cuda/sdk

all: path

gqd_qd_util.o: gqd_qd_util.h gqd_qd_util.cpp
	@-echo ">>> compiling utilities ..."
	g++ -O2 -I/usr/local/cuda/include -I$(GQD_HOME)/inc \
            -I$(QD_ROOT)/include -c gqd_qd_util.cpp
            
# D

workspace_host.o: workspace_host.h workspace_host.cpp
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c workspace_host.cpp -o workspace_host.o

eval_host.o: eval_host.h eval_host.cpp poly.o workspace_host.o
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c eval_host.cpp -o eval_host.o

path_host.o: path_host.h path_host.cpp predictor_host.o newton_host.o
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c path_host.cpp -o path_host.o

newton_host.o: newton_host.h newton_host.cpp eval_host.o mgs2_host.o 
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c newton_host.cpp -o newton_host.o

predictor_host.o: predictor_host.h predictor_host.cpp workspace_host.o
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include \
            -c predictor_host.cpp -o predictor_host.o

mgs2_host.o: mgs2_host.h mgs2_host.cpp 
	g++ -O2 -I./DefineTypesD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c mgs2_host.cpp -o mgs2_host.o

path_kernel.o: path_kernel.h path_kernel.cu gpu_data.h path_kernel_d.cu complex.h DefineTypesD/DefineType.h
	@-echo ">>> compiling kernels ..."
	nvcc -O2  --define-macro precision=d -I./DefineTypesD \
              -I$(GQD_HOME)/inc -I$(SDK_HOME)/C/common/inc \
             -I/usr/local/cuda/include -o path_kernel.o \
             -c path_kernel.cu --ptxas-options=-v

utilities.o: utilities.h utilities.cpp
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o utilities.o utilities.cpp

families.o: families.h families.cpp
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o families.o families.cpp

poly.o: poly.h poly.cpp utilities.o DefineTypesD/DefineType.h
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o poly.o poly.cpp
	         
path.o: path.cpp families.o path_host.o path_kernel.o newton_host.o parameter.h
	g++ -O2  -I./DefineTypesD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o path.o path.cpp
	         
path: families.o path.o poly.o utilities.o path_kernel.o gqd_qd_util.o path_host.o eval_host.o
	g++ -O2 -I$(GQD_HOME)/inc -I$(QD_ROOT)/include \
		path.o families.o poly.o utilities.o path_kernel.o gqd_qd_util.o\
		predictor_host.o mgs2_host.o newton_host.o path_host.o eval_host.o\
		workspace_host.o\
		$(QD_LIB)/libqd.a -o path \
		-lcutil_x86_64 -lcudart \
		-L/usr/local/cuda/lib64 -L$(SDK_HOME)/C/lib
            
# DD

workspace_host_dd.o: workspace_host.h workspace_host.cpp
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c workspace_host.cpp -o workspace_host_dd.o

eval_host_dd.o: eval_host.h eval_host.cpp poly_dd.o workspace_host_dd.o
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c eval_host.cpp -o eval_host_dd.o

path_host_dd.o: path_host.h path_host.cpp predictor_host_dd.o newton_host_dd.o
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c path_host.cpp -o path_host_dd.o

newton_host_dd.o: newton_host.h newton_host.cpp eval_host_dd.o mgs2_host_dd.o 
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c newton_host.cpp -o newton_host_dd.o

predictor_host_dd.o: predictor_host.h predictor_host.cpp workspace_host_dd.o
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include \
            -c predictor_host.cpp -o predictor_host_dd.o

mgs2_host_dd.o: mgs2_host.h mgs2_host.cpp 
	g++ -O2 -I./DefineTypesDD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c mgs2_host.cpp -o mgs2_host_dd.o

path_kernel_dd.o: path_kernel.h path_kernel.cu gpu_data.h path_kernel_dd.cu complex.h DefineTypesDD/DefineType.h
	@-echo ">>> compiling kernels ..."
	nvcc -O2  --define-macro precision=dd -I./DefineTypesDD \
              -I$(GQD_HOME)/inc -I$(SDK_HOME)/C/common/inc \
             -I/usr/local/cuda/include -o path_kernel_dd.o \
             -c path_kernel.cu --ptxas-options=-v

utilities_dd.o: utilities.h utilities.cpp
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o utilities_dd.o utilities.cpp

families_dd.o: families.h families.cpp
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o families_dd.o families.cpp

poly_dd.o: poly.h poly.cpp utilities_dd.o DefineTypesDD/DefineType.h
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o poly_dd.o poly.cpp
	         
path_dd.o: path.cpp families_dd.o path_host_dd.o path_kernel_dd.o newton_host_dd.o parameter.h
	g++ -O2  -I./DefineTypesDD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o path_dd.o path.cpp
	         
path_dd: families_dd.o path_dd.o poly_dd.o utilities_dd.o path_kernel_dd.o gqd_qd_util.o path_host_dd.o eval_host_dd.o
	g++ -O2 -I$(GQD_HOME)/inc -I$(QD_ROOT)/include \
		path_dd.o families_dd.o poly_dd.o utilities_dd.o path_kernel_dd.o gqd_qd_util.o\
		predictor_host_dd.o mgs2_host_dd.o newton_host_dd.o path_host_dd.o eval_host_dd.o\
		workspace_host_dd.o\
		$(QD_LIB)/libqd.a -o path_dd \
		-lcutil_x86_64 -lcudart \
		-L/usr/local/cuda/lib64 -L$(SDK_HOME)/C/lib            
# QD

workspace_host_qd.o: workspace_host.h workspace_host.cpp
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c workspace_host.cpp -o workspace_host_qd.o

eval_host_qd.o: eval_host.h eval_host.cpp poly_qd.o workspace_host_qd.o
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c eval_host.cpp -o eval_host_qd.o

path_host_qd.o: path_host.h path_host.cpp predictor_host_qd.o newton_host_qd.o
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c path_host.cpp -o path_host_qd.o

newton_host_qd.o: newton_host.h newton_host.cpp eval_host_qd.o mgs2_host_qd.o 
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c newton_host.cpp -o newton_host_qd.o

predictor_host_qd.o: predictor_host.h predictor_host.cpp workspace_host_qd.o
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include \
            -c predictor_host.cpp -o predictor_host_qd.o

mgs2_host_qd.o: mgs2_host.h mgs2_host.cpp 
	g++ -O2 -I./DefineTypesQD -I$(QD_ROOT)/include  -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c mgs2_host.cpp -o mgs2_host_qd.o

path_kernel_qd.o: path_kernel.h path_kernel.cu gpu_data.h  path_kernel_qd.cu complex.h DefineTypesQD/DefineType.h
	@-echo ">>> compiling kernels ..."
	nvcc -O2  --define-macro precision=qd -I./DefineTypesQD \
              -I$(GQD_HOME)/inc -I$(SDK_HOME)/C/common/inc \
             -I/usr/local/cuda/include -o path_kernel_qd.o \
             -c path_kernel.cu --ptxas-options=-v

utilities_qd.o: utilities.h utilities.cpp
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o utilities_qd.o utilities.cpp
families_qd.o: families.h families.cpp
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o families_qd.o families.cpp

poly_qd.o: poly.h poly.cpp utilities_qd.o DefineTypesQD/DefineType.h
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o poly_qd.o poly.cpp
	         
path_qd.o: path.cpp families_qd.o path_host_qd.o path_kernel_qd.o newton_host_qd.o parameter.h
	g++ -O2  -I./DefineTypesQD -I$(QD_ROOT)/include -I/usr/local/cuda/include \
            -I$(GQD_HOME)/inc -c -o path_qd.o path.cpp
	         
path_qd: families_qd.o path_qd.o poly_qd.o utilities_qd.o path_kernel_qd.o gqd_qd_util.o path_host_qd.o eval_host_qd.o
	g++ -O2 -I$(GQD_HOME)/inc -I$(QD_ROOT)/include \
		path_qd.o families_qd.o poly_qd.o utilities_qd.o path_kernel_qd.o gqd_qd_util.o\
		predictor_host_qd.o mgs2_host_qd.o newton_host_qd.o path_host_qd.o eval_host_qd.o\
		workspace_host_qd.o\
		$(QD_LIB)/libqd.a -o path_qd \
		-lcutil_x86_64 -lcudart \
		-L/usr/local/cuda/lib64 -L$(SDK_HOME)/C/lib

clean:
	rm -rf *.o path path_dd path_qd
