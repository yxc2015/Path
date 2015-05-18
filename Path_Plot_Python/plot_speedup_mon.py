from pylab import *

dim = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352]

cpu = [0.56,0.722,1.261,2.259,3.495,5.613,7.98,11.887,16.361,22.342,29.557, \
     37.941,47.841,59.65,73.386,89.924,106.965,127.49,149.941,175.21,202.515,232.267]

gpu_tree = [32.608, 45.121, 66.945, 94.209, 155.842, 228.35, 324.038, 444.165, 645.794, 877.899, 1156.628, 1473.873, \
     1831.143, 2237.872, 2689.535, 3187.989, 3948.801, 4802.077, 5731.47, 6748.997, 7847.115, 9024.241]

gpu_align = [48.513, 96.705, 148.292, 212.352, 290.952, 360.52, 472.106, 663.242, 888.78, 1133.004, 1464.76, 1754.924, \
    2275.02, 2765.924, 3365.404, 3860.988, 4776.6, 5588.8, 6552.16, 7298.305, 8750.556, 9997.332]

gpu_seq = [93.793, 205.44, 324.484, 473.738, 737.8, 1540.904, 2930.372, 4948.052, 7161.568, 10149.47, 13745.96, \
     18145.99, 23336.09, 29748.99, 36669.63, 44698.98, 54033.6, 64331.92, 76727.12, 89359.41, 103395.2, 119245.7]


gpu_tree_speedup = []
gpu_align_speedup  = []
gpu_seq_speedup  = []

for i in range(0,len(dim)):
    gpu_tree_speedup.append(cpu[i]/gpu_tree[i]*1000)
    gpu_align_speedup.append(cpu[i]/gpu_align[i]*1000)
    gpu_seq_speedup.append(cpu[i]/gpu_seq[i]*1000)

figure(figsize=(8, 4))
subplot(1,1,1)
#ax.plot(cpu, 'o')
plot(dim, gpu_tree_speedup, 'D', label='Tree')
plot(dim, gpu_align_speedup, 'o', label='Reverse Aligned')
plot(dim, gpu_seq_speedup, '*', label='Reverse')

xticks(dim,rotation=70)
xlabel('Dimension of Cyclic Polynomial System')
ylabel('Speedup')
xlim(0,dim[-1]+16)

legend(bbox_to_anchor=(1.0, 0.5))

savefig("aaa.eps",bbox_inches='tight', pad_inches=0.1)

show()