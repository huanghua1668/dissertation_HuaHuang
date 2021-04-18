import torch
import torch.utils.data
from torch.nn import functional as F

import numpy as np
import sklearn.datasets

from models import step, eval_step, csnn_learnable_r, pre_train

def vis(model, X_grid, xx, x_lin, y_lin, X_vis, mask, epoch, dir0, alpha=1., learnable_r = False):
    with torch.no_grad():
        if learnable_r:
            output = model(torch.from_numpy(X_grid).float(), alpha, model.r * model.r)
        else:
            output = model(torch.from_numpy(X_grid).float(), alpha, 1.)
        output = F.softmax(output, dim=1)
        confidence = output.max(1)[0].numpy()

    z = confidence.reshape(xx.shape)

    # plt.figure()
    # l = np.linspace(0.5, 1., 21)
    # plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l)  # , extend='both')
    # plt.colorbar()
    # plt.scatter(X_vis[mask, 0], X_vis[mask, 1], s=6, c='r')
    # plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1], s=6)
    # axs = plt.gca()
    # axs.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
    # axs.set_aspect('equal')
    # # dir0 = '/home/hh/data/moons/'
    # dir = dir0 + '/confidence_epoch_{}.png'.format(epoch)
    # plt.savefig(dir)

    dir = dir0 + '/moons_confidence_alpha1_epoch{}.npz'.format(epoch)
    np.savez(dir,  a=x_lin, b=y_lin, c=z, d=X_vis, e=mask)
    PATH = dir0 + '/csnn_epoch{}.pth'.format(epoch)
    stateDict = model.state_dict()
    torch.save(stateDict, PATH)

maxAlpha = 1.
num_classes = 2
batchSize = 64
features = 64
learningRate = 0.001
l2Penalty = 1.0e-3
# l2Penalty = 1.0e-4
runs = 1
seeds = [0]
r2 = 1.
LAMBDA = 0.0 # radius penalty effect
MIU = 0.0 # penalty of weight of CSNN, l2 norm coefficient
epochs = 501
epochOutputs = [0, 5, 10, 50, 100, 500]
outputDir='results/evolvement/'
learnable_r = True
BIAS = False # need bias in the 1st hidden layer for MLP to get a good result

# Moons
noise = 0.1
# sklearn has no random seed, it depends on numpy to get random numbers
np.random.seed(0)
x_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
x_train0 = x_train
x_validate, y_validate = sklearn.datasets.make_moons(n_samples=200, noise=noise)
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
print('mean, std', mean, std)
x_train = (x_train-mean)/std/np.sqrt(2)
x_validate = (x_validate-mean)/std/np.sqrt(2)

# dataset for image output
domain = 3
x_lin = np.linspace(-domain+0.5, domain+0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
x_lin = (x_lin-mean[0])/std[0]/np.sqrt(2)
y_lin = (y_lin-mean[1])/std[1]/np.sqrt(2)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])

X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
mask = y_vis.astype(np.bool)
X_vis = (X_vis-mean)/std/np.sqrt(2) # no need here, as contour grid is built on x_lin, y_lin

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate).float(),
                                         F.one_hot(torch.from_numpy(y_validate)).float())

accs = []
losses = []
accs_validate = []
losses_validate = []


# pre_train
for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    model = csnn_learnable_r(2, features, bias=BIAS)
    if learnable_r:
        model.set_lambda(LAMBDA)
        model.set_miu(MIU)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)
    accuracy, loss, accuracy_validate, loss_validate = pre_train(model, optimizer, dl_train, dl_test, x_train,
                                                                 y_train, x_validate, y_validate, run, outputDir,
                                                                 maxEpoch=50)
    accs.append(accuracy)
    losses.append(loss)
    accs_validate.append(accuracy_validate)
    losses_validate.append(loss_validate)
# dir = outputDir + 'pre_train_acc_loss_csnn.npz'
# np.savez(dir, a=np.mean(accs, axis=0), b=np.std(accs, axis=0),
#          c=np.mean(losses, axis=0), d=np.std(losses, axis=0),
#          e=np.mean(accs_validate, axis=0), f=np.std(accs_validate, axis=0),
#          g=np.mean(losses_validate, axis=0), h=np.std(losses_validate, axis=0))

ACCs = []
ACCs_val = []
LOSSs = []
LOSSs_val = []
ALPHAs = None
bestValidationAccs = []
# runs = 0
for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    PATH = outputDir + '/csnn_run{}_epoch{}.pth'.format(run, 0)
    l = torch.load(PATH)
    model = l['net']
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)

    losses = []
    accuracies = []
    losses_validate = []
    accuracies_validate = []
    alphas = []
    mmcs = []
    nzs = []
    aucs = []
    rs = []

    bestValidationAcc = 0.
    np.set_printoptions(precision=4)
    for epoch in range(epochs):
        # alpha = maxAlpha * epoch/epochs
        alpha = maxAlpha
        for i, batch in enumerate(dl_train):
            if learnable_r:
                loss_ce, loss_penalty, loss_l2, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2, learnable_r=True)
            else:
                loss, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2)

        if learnable_r:
            accuracy, loss_ce, loss_penalty, loss_l2 = eval_step(model, x_train, y_train, alpha, r2, learnable_r=True)
            testacc, testloss_ce, testloss_penalty, testloss_l2 = eval_step(model, x_validate, y_validate, alpha, r2, learnable_r=True)
        else:
            accuracy, loss = eval_step(model, x_train, y_train, alpha, r2)
            testacc, testloss = eval_step(model, x_validate, y_validate, alpha, r2)
        if testacc > bestValidationAcc:
            bestValidationAcc = testacc

        if epoch % 5 == 0:
            if learnable_r:
                losses.append(loss_ce+loss_penalty)
                losses_validate.append(testloss_ce+testloss_penalty)
            else:
                losses.append(loss)
                losses_validate.append(testloss)
            accuracies.append(accuracy)
            accuracies_validate.append(testacc)
            if learnable_r:
                rs.append([torch.norm(model.r, p=float('inf')).detach().item(), torch.norm(model.r, p=2).detach().item()])
                rNorm = (torch.norm(model.r, p=2)).detach().numpy()
                print('epoch {}, alpha {:.2f}, r2 {:.1f}, train {:.3f}, test {:.3f}, ||r|| {:.3f}'
                     .format(epoch, alpha, r2, accuracy,testacc, rNorm))
                print('loss: cross_entropy {:.4f}, r penalty {:.4f}, w penalty {:.4f}'.format(loss_ce, loss_penalty, loss_l2))
                # r = np.sort(model.r.detach().numpy())[-10:]
                # print('r top 10: ', r)
            else:
                print('epoch {}, alpha {:.2f}, r2 {:.1f}, train {:.3f}, test {:.3f}'
                   .format(epoch, alpha, r2, accuracy, testacc))
        if epoch in epochOutputs:
            vis(model, X_grid, xx, x_lin, y_lin, X_vis, mask, epoch, outputDir, alpha, learnable_r)
            # plot_circles(model.fc1, x_train, y_train, alpha, model.r.detach().numpy(), epoch, BIAS)

    # plot_save_loss(losses, losses_validate, outputDir+'/loss_run{}.png'.format(run))
    # plot_save_acc(accuracies, accuracies_validate, outputDir+'/acc_run{}.png'.format(run))
    # plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, aucs,
    #                        outputDir+'/acc_nzs_mmcs_run{}.png'.format(run))

    bestValidationAccs.append(max(accuracies_validate))
    # AUCs.append(aucs)
    ACCs.append(accuracies)
    ACCs_val.append(accuracies_validate)
    LOSSs.append(losses)
    LOSSs_val.append(losses_validate)
    if ALPHAs is None: ALPHAs = alphas

# AUCs = np.array(AUCs)
ACCs = np.array(ACCs)
ACCs_val = np.array(ACCs_val)
LOSSs = np.array(LOSSs)
LOSSs_val = np.array(LOSSs_val)
print('mean and std of best validation acc in {} runs: {:.4f}, {:.4f}'
      .format(runs, np.mean(np.array(bestValidationAccs)), np.std(np.array(bestValidationAccs))))
# dir = outputDir + '/mean_std_accs_aucs_net4.npz'
# np.savez(dir,# a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
#          c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
#          e=np.mean(ACCs_val, axis=0), f=np.std(ACCs_val, axis=0),
#          g=np.mean(LOSSs, axis=0), h=np.std(LOSSs, axis=0),
#          i=np.mean(LOSSs_val, axis=0), j=np.std(LOSSs_val, axis=0),
#          k=ALPHAs, l=np.array(rs))

