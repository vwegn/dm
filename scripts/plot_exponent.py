import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
plt.style.use('fivethirtyeight')
mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['savefig.pad_inches'] = 0.1

config = "2_Seq_d"
true_epoch_len = 1

ax = plt.gca()
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

exponents = np.load("../training_results/" + config + "/exponents.npy")
data = np.squeeze(exponents).T
x = range(1, exponents.shape[2]+1)

plot = plt.plot(x, data)
plt.xlim(xmin=1)
# plt.xlim(xmax=221*9)
# plt.ylim(ymax=8)
num_passings = math.floor((len(data) / true_epoch_len))
ticks = [i * true_epoch_len for i in range(0, num_passings+1, 10)]
plt.xticks(ticks)
plt.xlabel("epochs")
plt.ylabel("exponent value")#
# plt.title("Evolution of exponents")
#plt.xticks([1, 1471])

# fig = plt.gcf()
# ax = plt.gca()
# lgd = ax.legend()

plt.legend(iter(plot), [r'$\nu_{}$'.format(i) for i in range(1, exponents.shape[1]+1)])
plt.savefig("../training_results/" + config + "/exponents.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.show()
# Epoch 2942/2942
# All sequences processed. Skipped windows: 5784
# Total windows: 88060
# --- Runtime without imports: 8439.563656330109 seconds ---


# --- Runtime without imports: 8443.62460231781 seconds ---
