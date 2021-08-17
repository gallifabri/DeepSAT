import numpy as np
import matplotlib.pyplot as plt


path2 = "experiment_data/findingsB8_means.npy"
path3 = "experiment_data/findingsB7_means.npy"
path4 = "experiment_data/findingsB6_means.npy"


std_means = np.load(path2)

std_means2 = [np.mean(std_means[i:i+30]) for i in range(0,len(std_means))]

rnd_means = np.load(path3)

rnd_means2 = [np.mean(rnd_means[i:i+30]) for i in range(0,len(rnd_means))]

rnd_cut = rnd_means2[:len(std_means)]

grd_means = np.load(path4)
grd_means2 = [np.mean(grd_means[i:i+30]) for i in range(0,len(grd_means))]

grd_cut = grd_means2[:len(std_means)]




plotdir = "plots/theta"
#deep = plt.plot(std_means, rnd_cut, grd_cut, label="DeepSAT")

deep, = plt.plot(std_means2, label="False")
rand, = plt.plot(rnd_cut, label="True")
#gree, = plt.plot(grd_cut, label="gamma = 0.")

#plt.legend(handles=[deep])
plt.legend(handles=[deep, rand])
#plt.legend([deep, rand, gree], ['DeepSAT', 'Random', 'Greedy'])



plt.xlabel("Episodes")
plt.ylabel("Variables Guessed")
plt.title("Randomized Weights")


plt.savefig(plotdir)



# path = "experiment_data/findings2_numsteps.npy"
# path2 = "experiment_data/findings2_means.npy"
# path3 = "experiment_data/findings3_means.npy"
# path4 = "experiment_data/findings4_means.npy"


# std_rolls = np.load(path)
# std_means = np.load(path2)

# rnd_means = np.load(path3)
# rnd_cut = rnd_means[:len(std_means)]

# grd_means = np.load(path4)
# grd_cut = grd_means[:len(std_means)]




# plotdir = "plots/30sat"
# #deep = plt.plot(std_means, rnd_cut, grd_cut, label="DeepSAT")

# deep, = plt.plot(std_means, label="DeepSAT")
# rand, = plt.plot(rnd_cut, label="Random")
# gree, = plt.plot(grd_cut, label="Greedy")

# #plt.legend(handles=[deep])
# plt.legend(handles=[deep, rand, gree])
# #plt.legend([deep, rand, gree], ['DeepSAT', 'Random', 'Greedy'])



# plt.xlabel("Episodes")
# plt.ylabel("Variables Guessed")
# plt.title("30SAT")


# plt.savefig(plotdir)