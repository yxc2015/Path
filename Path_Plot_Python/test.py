import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

def read_until_line(f, line):
    while True:
        tmp_line = f.readline()
        if tmp_line[:len(line)] == line:
            break
    return

def read_complex_pt(f,dim):
    pt= []
    print "a"
    for _ in range(dim):
        # XXX
        tmp_line = f.readline()
        tmp_list = tmp_line.split(':')
        tmp_list = tmp_list[-1].split('  ')
        print tmp_list
        pt.append(complex(float(tmp_list[-2]),float(tmp_list[-1])))
    return pt
        
def print_pt(dim, pt):
    for i in range(dim):
        print "%2d  %+2.10e %+2.10e"%(i, pt[i].real, pt[i].imag)
        
def log_point(x):
    if x<0:
        return -sqrt(-x)
    else:
        return sqrt(x)


class correct_iteration:        
    def __init__(self, correct_a, correct_r, residual_a, residual_r):
        self.correct_a = correct_a
        self.correct_r = correct_r
        self.residual_a = residual_a
        self.residual_r = residual_a
        
    def show(self):
        print "correct(a&r) ", self.correct_a, self.correct_r, \
              " residue(a&r) ", self.residual_a, self.residual_r

class PathStep:
    def __init__(self,dim):
        self.dim = dim
        self.success = False
        self.t = complex(0,0)
        self.delta_t = 0
        self.predict_pt = []
        self.correct_pt = []
        self.correct_it = []
        self.n_it = 0
        
    def read_phc_file(self,f, tmp_line, inverse):
        print tmp_line
        tmp_list = tmp_line.split(' ')
        self.delta_t = float(tmp_list[3])
        #print tmp_list
        if(inverse==1):
            self.t = complex(1.0-float(tmp_list[8]), float(tmp_list[11]))
        else:
            self.t = complex(float(tmp_list[8]), float(tmp_list[11]))
            
        #f.readline()
        self.predict_pt = read_complex_pt(f, self.dim)
        #print self.predict_pt
        self.read_it(f)
        read_until_line(f, "the solution")
        self.correct_pt = read_complex_pt(f, self.dim)
    
    def read_it(self,f):
        while True:
            tmp_line = f.readline()
            tmp_list = tmp_line.split(" ")
            if tmp_list[0] == "correction":
                tmp_it = correct_iteration(tmp_list[3],tmp_list[5],tmp_list[9],tmp_list[11])
                self.correct_it.append(tmp_it)
                self.n_it += 1
            else:
                break
            
    def show(self):
        print "t = %.4e  delta_t = %.4e"%(self.t.real,self.delta_t)
        print_pt(self.dim, self.predict_pt)
        for tmp_it in self.correct_it:
            tmp_it.show()
        print_pt(self.dim, self.correct_pt)
        
    def value(self,var_idx):
        return self.t.real, self.correct_pt[var_idx].real, self.correct_pt[var_idx].imag, \
            self.predict_pt[var_idx].real, self.predict_pt[var_idx].imag
    
    
class Path:
    def __init__(self, dim, inverse=0):
        self.dim = dim
        self.n_step = 0
        self.steps = []
        self.pt_idxs = []
        self.start_pt = []
        self.inverse = inverse
        
    def read_phc_file(self,f):
        output_symbol = "OUTPUT INFORMATION"
        read_until_line(f, output_symbol)
        read_until_line(f, "the solution")
        self.start_pt = read_complex_pt(f, self.dim)
        
        step_end_symbol = "== err : "
        step_start_symbol = "step "
        corrector_symbol = "correction (a&r)"
        
        while True:
            read_until_line(f, step_end_symbol)
            tmp_line = f.readline()
            print tmp_line
            if(tmp_line[:len(step_start_symbol)] != step_start_symbol):
                if(tmp_line[:len(corrector_symbol)] == corrector_symbol):
                    read_until_line(f, step_end_symbol);
                    read_until_line(f, step_end_symbol);
                    tmp_line = f.readline()
                    if(tmp_line[:len(step_start_symbol)] != step_start_symbol):
                        break;
                else:
                    break
            self.add_step(f, tmp_line)
            
        self.check_success_pts()
        #print self.pt_idxs
        return
    
    def check_success_pts(self):
        for i in range(self.n_step-1):
            if(self.inverse==0):
                if (self.steps[i+1].t.real > self.steps[i].t.real):
                    self.pt_idxs.append(i)
            else:
                if (self.steps[i+1].t.real < self.steps[i].t.real):
                    self.pt_idxs.append(i)
        
        if self.inverse == False:
            if self.steps[self.n_step-1].t.real == 1.0:
                    self.pt_idxs.append(self.n_step-1)
        else:
            if self.steps[self.n_step-1].t.real == 0.0:
                    self.pt_idxs.append(self.n_step-1)
                    
    
    def add_step(self,f, tmp_line):
        b = PathStep(self.dim)
        self.n_step += 1
        b.read_phc_file(f, tmp_line, self.inverse)
        self.steps.append(b)
        
    def show(self):
        for i in range(self.n_step):
            print "step", i
            self.steps[i].show()
            
    def array_3d(self, start_t=0.0, end_t=1.0, var_idx=0):
        ts = []
        xs = []
        ys = []
        ts_predict = []
        xs_predict = []
        ys_predict = []
        ts_predict_fail = []
        xs_predict_fail = []
        ys_predict_fail = []
        
        if(self.inverse==0):
            t =0.0
        else:
            t =1.0
            
        use_log_point = False
        if(t>=start_t and t<=end_t):
            ts.append(t)
            x = self.start_pt[var_idx].real
            y = self.start_pt[var_idx].imag
            if(use_log_point==True):
                x = log_point(x)
                y = log_point(y)
            xs.append(x)
            ys.append(y)
            
        for i in range(self.n_step):
            t, x, y, x_predict, y_predict = self.steps[i].value(var_idx)
            if(use_log_point==True):
                x = log_point(x)
                y = log_point(y)
                x_predict = log_point(x_predict)
                y_predict = log_point(y_predict)
                
            if(t>=start_t and t<=end_t):
                ts_predict.append(t)
                xs_predict.append(x_predict)
                ys_predict.append(y_predict)
                if i in self.pt_idxs:
                    ts.append(t)
                    xs.append(x)
                    ys.append(y)
                else:
                    ts_predict_fail.append(t)
                    xs_predict_fail.append(x_predict)
                    ys_predict_fail.append(y_predict)
                          
        return ts, xs, ys, \
            ts_predict, xs_predict, ys_predict, \
            ts_predict_fail, xs_predict_fail, ys_predict_fail
            
    def var_range(self, var_idx):
        real_min = self.start_pt[var_idx].real
        real_max = self.start_pt[var_idx].real
        imag_max = self.start_pt[var_idx].imag
        imag_min = self.start_pt[var_idx].imag
        for i in range(self.n_step):
            if self.steps[i].predict_pt[var_idx].real < real_min:
                real_min = self.steps[i].predict_pt[var_idx].real
            elif self.steps[i].predict_pt[var_idx].real > real_max:
                real_max = self.steps[i].predict_pt[var_idx].real
            if self.steps[i].predict_pt[var_idx].imag < imag_min:
                imag_min = self.steps[i].predict_pt[var_idx].imag
            elif self.steps[i].predict_pt[var_idx].imag > imag_max:
                imag_max = self.steps[i].predict_pt[var_idx].imag
        return real_min, real_max, imag_max, imag_min

def plot_path(path_data, plot_predict=True, var_idx=0, var_range=[0,0,0,0]):
    n_path = len(path_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x-real')
    ax.set_ylabel('x-imag')
    ax.set_zlabel('t')
    for i in range(0,n_path):
        ts, xs, ys, ts_predict, xs_predict, ys_predict, ts_predict_fail, xs_predict_fail, ys_predict_fail = path_data[i].array_3d(var_idx=var_idx)
        ax.plot(xs, ys, ts, 'b')
        if(i==n_path-1):
            ax.plot(xs, ys, ts, 'b*', markersize=5, label='correct')           
        if(plot_predict == True):
            ax.plot(xs_predict, ys_predict, ts_predict, 'r.', markersize=6, label='predict')
            ax.plot(xs_predict_fail, ys_predict_fail, ts_predict_fail, 'bo', \
                    mfc='none', markersize=6, label='divergent')
        ax.plot([xs[0]], [ys[0]], [ts[0]], 'bo', markersize=6)
        if(i==n_path-1):
            ax.plot([xs[-1]], [ys[-1]], [ts[-1]], 'ro', mfc='none', markersize=10)
    #ax.legend(bbox_to_anchor=(0.9, 0.9))
    ax.set_zlim([0,1])
    ax.set_xlim([var_range[0],var_range[1]])
    ax.set_ylim([var_range[2],var_range[3]])
    plt.gca().invert_zaxis()
    ax.view_init(elev=20., azim=-135)
    #ax.view_init(elev=30., azim=45)

def compare_path(path_data1, path_data2, plot_predict=True, start_t=0.0, end_t=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ts, xs, ys, ts_predict, xs_predict, ys_predict, ts_predict_fail, xs_predict_fail, ys_predict_fail = path_data1.array_3d(start_t, end_t)
    print len(ts)
    ax.plot(xs, ys, ts, 'g.')
    ax.plot(xs, ys, ts, 'b')
    if(plot_predict == True):
        ax.plot(xs_predict, ys_predict, ts_predict, 'r.')
        ax.plot(xs_predict_fail, ys_predict_fail, ts_predict_fail, 'bo', mfc='none')
    ts, xs, ys, ts_predict, xs_predict, ys_predict, ts_predict_fail, xs_predict_fail, ys_predict_fail = path_data2.array_3d(start_t, end_t)
    print len(ts)
    ax.plot(xs, ys, ts, 'r.')
    ax.plot(xs, ys, ts, 'gold')
    if(plot_predict == True):
        ax.plot(xs_predict, ys_predict, ts_predict, 'r.')
        ax.plot(xs_predict_fail, ys_predict_fail, ts_predict_fail, 'bo', mfc='none')
    ax.set_zlim([start_t,end_t])
    plt.gca().invert_zaxis()
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    plt.show()



"""
path_list=[]
dim = 10
f = open("../../Problems/test/cyclic10.path","r")
n_path = 1
for i in range(n_path):
    path_data = Path(dim,0)
    path_data.read_phc_file(f)
    path_list.append(path_data)
new_path_list = [] 
path_idx = range(0,n_path)
#for i in path_idx:
#    new_path_list.append(path_list[i])

new_path_list.append(path_list[0])
plot_path(new_path_list[:1], True, var_idx=1)        
#plt.show()
plt.savefig("test.eps",bbox_inches='tight', pad_inches=0.12)

"""
path_list=[]
dim = 4
f = open("../../Problems/test/cyclic4.path","r")
n_path = 10
for i in range(n_path):
    path_data = Path(4,i%2)
    path_data.read_phc_file(f)
    path_list.append(path_data)
new_path_list = [] 
path_idx = range(0,n_path)
for i in path_idx:
    new_path_list.append(path_list[i])

plot_var = 1
all_real_min, all_real_max, all_imag_max, all_imag_min=new_path_list[0].var_range(plot_var)
for i in range(1,len(new_path_list)):
    real_min, real_max, imag_max, imag_min=new_path_list[i].var_range(plot_var)
    print real_min, real_max, imag_max, imag_min
    if real_min < all_real_min:
        all_real_min = real_min
    if real_max > all_real_max:
        all_real_max = real_max
    if imag_min < all_imag_min:
        all_imag_min = imag_min
    if imag_max > all_imag_max:
        all_imag_max = imag_max

print all_real_min, all_real_max, all_imag_max, all_imag_min
    
for i in range(0,len(new_path_list)):
    plot_path(new_path_list[:i+1], False, var_idx=plot_var, var_range=[all_real_min, all_real_max, all_imag_max, all_imag_min])
    plt.savefig("monodromy%s.eps"%i,bbox_inches='tight', pad_inches=0.0)
plt.show()
"""
path_list=[]
dim = 16
f = open("../../Problems/test/cyclic16.path","r")
n_path = 10
for i in range(0,n_path):
    path_data = Path(dim,i%2)
    path_data.read_phc_file(f)
    path_list.append(path_data)
new_path_list = [] 
path_idx = range(0,n_path)
for i in path_idx:
    new_path_list.append(path_list[i])
for i in range(0,len(new_path_list)):
    #for j in range(0,dim):
    plot_path(new_path_list[:i+1], True, var_idx=0)
plt.show()
"""
"""
f = open("../../Problems/test/cyclic10.path","r")
path_data = Path(10)
path_data.read_phc_file(f)
#path_data.show()


f = open("../../Problems/test/cyclic10-1.path","r")
path_data1 = Path(10)
path_data1.read_phc_file(f)
#path_data1.show()

for i in range(0,10):
    plot_path([path_data], var_idx=i)
plt.show()

"""
#plot_path([path_data,path_data1],False)
#compare_path(path_data,path_data1,False)