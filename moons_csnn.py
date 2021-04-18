import torch.utils.data
from torch.nn import functional as F

import numpy as np
import sklearn.datasets

from sklearn.metrics import roc_curve, auc

from utils.plot_utils import plot_distribution, plot_save_roc, plot_save_acc_nzs_auroc
from utils.data_preprocess import get_threshold, extract_ood
from models import nnz, step, eval_step, eval_combined, csnn_learnable_r, pre_train

#def vis(model, X_grid, xx, x_lin, y_lin, X_vis, mask, epoch, dir0, alpha=1., learnable_r = False):
#    with torch.no_grad():
#        if learnable_r:
#            output = model(torch.from_numpy(X_grid).float(), alpha, model.r * model.r)
#        else:
#            output = model(torch.from_numpy(X_grid).float(), alpha, 1.)
#        output = F.softmax(output, dim=1)
#        confidence = output.max(1)[0].numpy()
#
#    z = confidence.reshape(xx.shape)
#
#    # plt.figure()
#    # l = np.linspace(0.5, 1., 21)
#    # plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l)  # , extend='both')
#    # plt.colorbar()
#    # plt.scatter(X_vis[mask, 0], X_vis[mask, 1], s=6, c='r')
#    # plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1], s=6)
#    # axs = plt.gca()
#    # axs.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
#    # axs.set_aspect('equal')
#    # # plt.axis([-3, 3., -3, 3])
#    # # dir0 = '/home/hh/data/moons/'
#    # dir = dir0 + '/confidence_epoch_{}.png'.format(epoch)
#    # plt.savefig(dir)
#    # # plt.show()
#    # if(epoch == 190):
#    #     dir = dir0 + '/moons_confidence_alpha0.npz'
#    #     np.savez(dir,  a=x_lin, b=y_lin, c=z, d=X_vis, e=mask)

# seeds = [0, 100057, 300089, 500069, 700079]
num_classes = 2
batchSize = 64
features = 64
learningRate = 0.001
l2Penalty = 1.0e-3
runs = 1
r2 = 1.
maxAlpha = 1.
LAMBDA = 0.64
# LAMBDA = 0.004
MIU = 0.0
epochs = 201
outputDir='results/'
learnable_r = True
BIAS = False
percentage = 0.99

# Moons
# noise = 0.1 # for visual demonstration of effectiveness of CSNN, use 0.1
noise = 0.18 # for OOD detection, enlarge it to 0.18 to make seperation harder
# sklearn has no random seed, it depends on numpy to get random numbers
np.random.seed(0)
torch.manual_seed(0)
x_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
x_train0 = x_train
x_test, y_test = sklearn.datasets.make_moons(n_samples=500, noise=noise)
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
print('mean, std', mean, std)
x_train = (x_train-mean)/std/np.sqrt(2)
x_test = (x_test-mean)/std/np.sqrt(2)

# dataset for image output
domain = 3
x_lin = np.linspace(-domain+0.5, domain+0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
x_lin = (x_lin-mean[0])/std[0]/np.sqrt(2)
y_lin = (y_lin-mean[1])/std[1]/np.sqrt(2)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])

minDis, oodLabel = get_threshold(x_train, percentage, outputDir)
x_inDis = x_train[~oodLabel]
y_inDis = y_train[~oodLabel]
mask, x_ood = extract_ood(x_inDis, X_grid, minDis)

dir = outputDir + 'moons_train_test_ood.npz'
np.savez(dir, a=x_inDis, b=y_inDis, c=x_test, d=y_test, e=x_ood)

x_combined = np.concatenate((x_test, x_ood))
print('in-dis {}, ood {}'.format(x_test.shape[0], x_ood.shape[0]))

label_ood = np.zeros(x_combined.shape[0])
label_ood[x_test.shape[0]:] = 1

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(),
                                         F.one_hot(torch.from_numpy(y_test)).float())
ds_combined = torch.utils.data.TensorDataset(torch.from_numpy(x_combined).float())

# pre_train
# accs = []
# losses = []
# accs_test = []
# losses_test = []
for run in range(runs):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_test.shape[0], shuffle=False)
    model = csnn_learnable_r(2, features, bias=BIAS)
    if learnable_r:
        model.set_lambda(LAMBDA)
        model.set_miu(MIU)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)
    accuracy, loss, accuracy_test, loss_test = pre_train(model, optimizer, dl_train, dl_test, x_train,
                                                                 y_train, x_test, y_test, run, outputDir,
                                                                 maxEpoch=50)
#     accs.append(accuracy)
#     losses.append(loss)
#     accs_test.append(accuracy_test)
#     losses_test.append(loss_test)
# dir = outputDir + 'pre_train_acc_loss_csnn.npz'
# np.savez(dir, a=np.mean(accs, axis=0), b=np.std(accs, axis=0),
#          c=np.mean(losses, axis=0), d=np.std(losses, axis=0),
#          e=np.mean(accs_test, axis=0), f=np.std(accs_test, axis=0),
#          g=np.mean(losses_test, axis=0), h=np.std(losses_test, axis=0))

ACCs = []
ACCs_test = []
LOSSs = []
LOSSs_test = []
AUCs = []
ALPHAs = None
for run in range(runs):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_test.shape[0], shuffle=False)
    dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
    PATH = outputDir + '/csnn_run{}_epoch{}.pth'.format(run, 0)
    l = torch.load(PATH)
    model = l['net']
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)

    losses = []
    accuracies = []
    losses_test = []
    accuracies_test = []
    alphas = []
    mmcs = []
    nzs = []
    aucs = []
    rs = []

    np.set_printoptions(precision=4)
    for epoch in range(epochs):
        # alpha = maxAlpha * (epoch+1)/epochs
        alpha = maxAlpha
        for i, batch in enumerate(dl_train):
            if learnable_r:
                loss_ce, loss_penalty, loss_l2, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2, learnable_r=True)
            else:
                loss, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2)

        if learnable_r:
            accuracy, loss_ce, loss_penalty, loss_l2 = eval_step(model, x_train, y_train, alpha, r2, learnable_r=True)
            testacc, testloss_ce, testloss_penalty, testloss_l2 = eval_step(model, x_test, y_test, alpha, r2, learnable_r=True)
        else:
            accuracy, loss = eval_step(model, x_train, y_train, alpha, r2)
            testacc, testloss = eval_step(model, x_test, y_test, alpha, r2)

        if epoch % 5 == 0:
            if learnable_r:
                losses.append(loss_ce+loss_penalty)
                losses_test.append(testloss_ce+testloss_penalty)
            else:
                losses.append(loss)
                losses_test.append(testloss)
            accuracies.append(accuracy)
            accuracies_test.append(testacc)

            if learnable_r:
                nz, mmc = nnz(x_ood, model, alpha, model.r * model.r)
            else:
                nz, mmc = nnz(x_ood, model, alpha, r2)

            alphas.append(epoch)
            mmcs.append(mmc)
            nzs.append(nz)
            uncertainties = eval_combined(model, dl_combined, alpha, r2, learnable_r)
            falsePositiveRate, truePositiveRate, _ = roc_curve(label_ood, -uncertainties)
            AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
            aucs.append(AUC)

            if learnable_r:
                rs.append([torch.norm(model.r, p=float('inf')).detach().item(), torch.norm(model.r, p=2).detach().item()])
                rNorm = (torch.norm(model.r, p=2)).detach().numpy()
                w = model.fc1.weight.data.numpy()
                r = model.r.detach().numpy()
                radius2 = r * r + np.sum(w * w, axis=1) * (1 / alpha / alpha - 1)
                if BIAS:
                    w0 = model.fc1.bias.detach().numpy()
                    radius2 += w0 * w0 * (1 / alpha / alpha - 1)
                print('loss: cross_entropy {:.4f}, r penalty {:.4f}, w penalty {:.4f}'.format(loss_ce, loss_penalty, loss_l2))
            print('epoch {}, alpha {:.2f}, r2 {:.1f}, nz {:.3f}, train {:.3f}, test {:.3f}, auroc {:.3f}, ||r||2 {:.3f}'
                  .format(epoch, alpha, r2, 1. - nz, accuracy, testacc, AUC, rNorm))
        # if epoch%10 == 0 or epoch<10:
        #    vis(model, X_grid, xx, x_lin, y_lin, X_vis, maskVis, epoch, outputDir, alpha, learnable_r)
            # plot_circles(model.fc1, x_train, y_train, alpha, model.r.detach().numpy(), epoch, outputDir, BIAS)
        if epoch == epochs-1:
            dir = outputDir + '/moons_hist_confidence.npz'
            np.savez(dir, a=uncertainties, b=x_test.shape[0], c=epochs-1)
            dir = outputDir + '/moons_roc.npz'
            np.savez(dir, a=falsePositiveRate, b=truePositiveRate, c=AUC)

    # plot_save_loss(losses, losses_test, outputDir+'/loss_run{}.png'.format(run))
    # plot_save_acc(accuracies, accuracies_test, outputDir+'/acc_run{}.png'.format(run))
    if run == 0:
        dir = outputDir + '/moons_acc_nzs_auroc.npz'
        np.savez(dir, a=alphas, b=accuracies_test, c=nzs, d=aucs)

    AUCs.append(aucs)
    ACCs.append(accuracies)
    ACCs_test.append(accuracies_test)
    LOSSs.append(losses)
    LOSSs_test.append(losses_test)
    if ALPHAs is None: ALPHAs = alphas

# AUCs = np.array(AUCs)
# ACCs = np.array(ACCs)
# ACCs_test = np.array(ACCs_test)
# LOSSs = np.array(LOSSs)
# LOSSs_test = np.array(LOSSs_test)
# dir = outputDir + '/moons_csnn_mean_std_accs_aucs.npz'
# np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
         # c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
         # e=np.mean(ACCs_test, axis=0), f=np.std(ACCs_test, axis=0),
         # g=np.mean(LOSSs, axis=0), h=np.std(LOSSs, axis=0),
         # i=np.mean(LOSSs_test, axis=0), j=np.std(LOSSs_test, axis=0),
         # k=ALPHAs, l=np.array(rs))

