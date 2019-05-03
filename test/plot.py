from matplotlib import pyplot as plt
import numpy as np

# a9a = np.array([1, 1.803005008, 3.176470588, 5.510204082, 8.925619835])
# cod_rna = np.array([1,	1.872355186,	3.240096038,	5.536410256,	8.907590759])
# ijcnn1 = np.array([1,	1.667803215,	3.158671587,	5.558441558,	8.84754522])
# webspam = np.array([1,	1.828194014,	3.342995169,	5.478476002,	10.59521531])
# phishing = np.array([1,	1.934593023,	3.352644836,	6.278301887,	10.23846154])
# w8a = np.array([1,	1.757307589,	2.9359319,	5.548687553,	10.15968992])
#
# ideal =  np.array([1,2,4,8,16])
# cores = [1,2,4,8,16]
# fig, ax = plt.subplots()
# ax.plot(cores, a9a, label='a9a')
# ax.plot(cores, cod_rna, label='cod_rna')
# ax.plot(cores, ijcnn1, label='ijcnn1')
# ax.plot(cores, webspam, label='webspam')
# ax.plot(cores, phishing, label='phishing')
# ax.plot(cores, w8a, label='w8a')
# ax.plot(cores, ideal, label='Ideal')
# legend = ax.legend(loc='upper right', shadow=True, fontsize='xx-small')
#
# plt.xlabel('cores')
# plt.ylabel('Speed Up')
# plt.title('Speed Up vs Cores')
# plt.show()


#a9a = np.array([1, 1.803005008, 3.176470588, 5.510204082, 8.925619835])
#cod_rna = np.array([1,	1.872355186,	3.240096038,	5.536410256,	8.907590759])
#ijcnn1 = np.array([1,	1.667803215,	3.158671587,	5.558441558,	8.84754522])
#webspam = np.array([1,	1.61528361,	2.853808771,	4.997688509,	7.343883485])
#phishing = np.array([1,	1.934593023,	3.352644836,	6.278301887,	10.23846154])
#w8a = np.array([1,	1.757307589,	2.9359319,	5.548687553,	10.15968992])
webspam_py = np.array([1,	1.913528743,	3.75819667,	7.409385514,	12.64104837])
webspam_java = np.array([1,	1.38446411,	1.916391211,	2.782608696,	3.862068966])
#webspam_c = np.array([1,	1.279020979,	1.812685828,	2.685756241,	4.11011236])

ideal =  np.array([1,2,4,8,16])
cores = [1,2,4,8,16]
fig, ax = plt.subplots()
#ax.plot(cores, a9a, label='a9a')
#ax.plot(cores, cod_rna, label='cod_rna')
#ax.plot(cores, ijcnn1, label='ijcnn1')
ax.plot(cores, webspam_py, label='python')
ax.plot(cores, webspam_java, label='java')
#ax.plot(cores, webspam_c, label='java')
#ax.plot(cores, phishing, label='phishing')
#ax.plot(cores, w8a, label='w8a')
ax.plot(cores, ideal, label='Ideal')
legend = ax.legend(loc='upper right', shadow=True, fontsize='xx-small')

plt.xlabel('Cores')
plt.ylabel('Speed Up')
plt.title('Single Node Core Level Speed Up - [Covtype, Split:0.80, 510K, 54F]')
plt.show()
