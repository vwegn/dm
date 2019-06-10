import itertools
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
plt.style.use('fivethirtyeight')
mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['savefig.pad_inches'] = 0.1

# mpl.rcParams['figure.facecolor'] = 'white'
# mpl.rcParams['patch.facecolor'] = 'white'
# mpl.rcParams['figure.figsize'] = 10, 8
#plt.style.use('seaborn-paper')


# Change Config!
config = "2_Seq_c"
true_epoch_len = 1
smoothing_param = 30 # math.floor(221 / 2)

history = np.load("../training_results/" + config + "/history.npy").item()

loss_reg = history['loss']
loss_unreg = history['unreg_loss_loss']
acc_2 = history['matches_accuracy_metric_2']
acc_5 = history['matches_accuracy_metric_5']
acc_10 = history['matches_accuracy_metric_10']
epe = history['matches_end_point_error_metric']

max_loss = max(max(loss_reg), max(loss_unreg))+0.1
min_loss = min(min(loss_reg), min(loss_unreg))-0.1
max_acc = max(max(acc_2), max(acc_5), max(acc_10))+0.004
min_acc = min(min(acc_2), min(acc_5), min(acc_10))-0.1



# N = 100
# x = range(0, len(acc))
# smooth_acc_1 = pd.Series(acc[:true_epoch_len]).rolling(window=N).mean().values
# smooth_acc_2 = pd.Series(acc[true_epoch_len:]).rolling(window=N).mean().values

# plot_acc = plt.plot(x, acc, linewidth=1)
# plot_acc = plt.plot(x[:true_epoch_len], smooth_acc_1)
# plot_acc = plt.plot(x[true_epoch_len:], smooth_acc_2, color=plot_acc[0].get_color())
# plt.xticks([0, true_epoch_len, 2942])
# plt.ylabel("accuracy@3")#
# plt.xlabel("epochs")

#


def plot_accuracy(data, smoothing):
    epochs = len(data)
    x = range(1, epochs)

    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')


    num_passings = math.floor(epochs / true_epoch_len)
    ticks = [i * true_epoch_len for i in range(0, num_passings+1, 10)]
    plt.xticks(ticks)
    plot = plt.plot(range(1, epochs+1), data, linewidth=3)


    for i in range(0, num_passings):
        s = pd.Series(data[true_epoch_len*i:true_epoch_len*(i+1)]).rolling(window=smoothing).mean().values
        plt.plot(range(true_epoch_len*i, true_epoch_len*(i+1)), s, color="red")







# plot = plt.plot(range(0, len(loss)), loss, linewidth=1)

# PLOT: regularized loss
plot_accuracy(loss_reg, smoothing_param)
plt.ylabel("loss (regularized)")
plt.xlabel("epochs")
# plt.xlim(xmin=1)
# plt.ylim(ymin=min_loss)
# plt.ylim(ymax=max_loss)


plt.savefig("../training_results/" + config + "/regularized_loss.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.clf()

# PLOT: unregularized loss
plot_accuracy(loss_unreg, smoothing_param)
plt.ylabel("loss")
plt.xlabel("epochs")
# plt.xlim(xmin=1)
# plt.ylim(ymin=min_loss)
# plt.ylim(ymax=max_loss)
plt.savefig("../training_results/" + config + "/unregularized_loss.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.clf()

# PLOT: accuracy@2
plot_accuracy(acc_2, smoothing_param)
plt.ylabel("accuracy@2")
plt.xlabel("epochs")
# plt.xlim(xmin=1)
# plt.ylim(ymin=0.918)
# plt.ylim(ymax=0.924)
# plt.ylim(ymin=0.935)
# plt.ylim(ymax=0.9405)
plt.savefig("../training_results/" + config + "/train_acc_2.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.clf()

# PLOT: accuracy@5
plot_accuracy(acc_5, smoothing_param)
plt.ylabel("accuracy@5")
plt.xlabel("epochs")
# plt.xlim(xmin=1)
# plt.ylim(ymin=0.969)
# plt.ylim(ymax=0.974)
# plt.ylim(ymin=0)
# plt.ylim(ymax=1)

plt.savefig("../training_results/" + config + "/train_acc_5.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.clf()


# PLOT: accuracy@10
plot_accuracy(acc_10, smoothing_param)
plt.ylabel("accuracy@10")
plt.xlabel("epochs")
# plt.xlim(xmin=1)
# plt.ylim(ymin=0.979)
# plt.ylim(ymax=0.985)
# plt.ylim(ymin=0.987)
# plt.ylim(ymax=0.9888)
plt.savefig("../training_results/" + config + "/train_acc_10.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.clf()

# PLOT: end-point-error
plot_accuracy(epe, smoothing_param)
plt.ylabel("end-point-error")
plt.xlabel("epochs")
# plt.xlim(xmin=1)
plt.savefig("../training_results/" + config + "/train_epe.pdf", dpi=300, format='pdf', bbox_inches='tight')
plt.clf()

# plot = plt.plot(x, loss_reg)
# plt.plot()#
# plt.axvline(true_epoch_len, color='black', linewidth=2)
# plt.xlim(xmin=1.0)
# plt.xlim(xmax=2942.0)
# plt.title("Evolution of exponents")


# plot_loss = plt.plot(x, loss_reg)

# fig = plt.gcf()
# ax = plt.gca()
# lgd = ax.legend()

# plt.legend(iter(plot), [r'$\nu_{}$'.format(i) for i in range(1, exponents.shape[1]+1)])

# Epoch 2942/2942
# All sequences processed. Skipped windows: 5784
# Total windows: 88060
# --- Runtime without imports: 8439.563656330109 seconds ---

# Total windows: 352240
#
# --- Runtime without imports: 34036.670357227325 seconds ---