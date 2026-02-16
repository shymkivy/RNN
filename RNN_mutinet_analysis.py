# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:01:34 2023

@author: ys2605
"""

import sys
import os

for user1 in ['ys2605', 'shymk']:
    if os.path.isdir('C:/Users/' + user1):
        path1 = 'C:/Users/' + user1 + '/Desktop/stuff/RNN_scripts/'

sys.path.append(path1)
sys.path.append(path1 + '/functions')


from f_analysis import f_plot_rates2, f_plot_rates_only # seriation, 
from f_RNN import f_RNN_test, f_RNN_test_spont, f_gen_ob_dset, f_gen_cont_dset, f_RNN_load_multinet, f_smooth_loss, f_data_quality_check #, f_trial_ave_pad, f_gen_equal_freq_space
from f_RNN_process import f_trial_ave_ctx_pad, f_trial_ave_ctx_pad2, f_trial_sort_data_pad, f_trial_sort_data_ctx_pad, f_label_redundants, f_get_rdc_trav, f_gather_dev_trials, f_gather_red_trials, f_gather_cont_trials, f_gather_cont_trials2d, f_analyze_rd_trial_vectors, f_analyze_cont_trial_vectors, f_get_trace_tau, f_get_tr_ave_tau, f_plot_t, f_get_stim_on_bins, f_get_diags_data, f_get_cs_same_dev, f_get_cs_same_red, f_get_cs_cont_to_ref_same_dev, f_get_cs_cont_to_ref_same_red, f_get_cs_cont_to_dev_vs_red, f_euc_dist, f_cos_sim, f_get_rd_vec_mags, f_get_cont_vec_mags, f_get_rd_vec_mags2, f_get_cont_vec_mags2
from f_RNN_dred import f_run_dred, f_run_dred_wrap, f_proj_onto_dred
from f_RNN_plots import f_plot_dred_rates, f_plot_dred_rates2, f_plot_dred_rates3, f_plot_dred_rates3d, f_plot_traj_speed, f_plot_resp_distances, f_plot_mmn, f_plot_mmn2, f_plot_mmn_dist, f_plot_mmn_freq, f_plot_dred_pcs, f_plot_rnn_weights2, f_plot_run_dist, f_plot_cont_vec_data, f_plot_rd_vec_data, f_plot_ctx_vec_data, f_plot_ctx_vec_dir, f_plot_cat_data, f_plot_cat_data2, f_plot_cat_data_violin, f_plot_cat_data_bar, f_plot_shadederrorbar2 # , f_plot_shadederrorbar
from f_RNN_chaotic import RNN_chaotic
from f_RNN_utils import f_gen_stim_output_templates, f_gen_cont_seq, f_gen_oddball_seq, f_gen_input_output_from_seq, f_plot_examle_inputs, f_plot_train_loss, f_plot_train_test_loss, f_gen_name_tag, f_cut_reshape_rates_wrap, f_plot_exp_var, f_plot_freq_space_distances_control, f_plot_freq_space_distances_oddball, f_save_fig # , f_reshape_rates
from f_RNN_decoder import f_make_cv_groups, f_sample_trial_data_dec, f_run_binwise_dec, f_plot_binwise_dec, f_run_one_shot_dec, f_plot_one_shot_dec_bycat, f_plot_one_shot_dec_bycat2, f_plot_one_shot_dec_iscat, f_plot_one_shot_dec_avecat

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from matplotlib import gridspec
#from matplotlib import colors
import matplotlib.cm as cm
#from random import sample, random
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import Isomap
#from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform, cdist #
from scipy.signal import correlate
#from scipy.sparse import diags
#from scipy import signal
from scipy import linalg, fft
#from scipy.io import loadmat, savemat
#import skimage.io

from datetime import datetime

#%%

#fname = 'test_data_2024_3_19_18h_6m'
# bad_ob = [1, 3,                     # short
#           4, 5, 13, 21,             # explode
#           8,                        # 250 neurons
#           #7, 9, 10, 11, 12, 14      # final loss too high
#           ] # 1 3 5 are short; 4 5 explode; 13 21 explode; 8 has 250 neurons; 10,11,12 not low enough

# fname = 'test_data_2024_5_23_11h_11m'
# bad_ob = [7]

data_path = 'F:/RNN_stuff/RNN_data/'

fname = 'test_data_2024_5_24_9h_42m'
bad_ob = []

bad_freq = []

fig_path = 'F:/RNN_stuff/fig_save/'

dred_subtr_mean = 0
dred_met = 2
num_skip_trials = 90
bad_rnn = [bad_ob, bad_freq, bad_ob]

nnet_plot = [0]


rnn_leg = ['ob trained', 'freq trained', 'untrained']
rnn_color = ['tab:blue', 'tab:orange', 'tab:green']

cmap1 = 'viridis'


#%%

test_ob_load = np.load(data_path + 'test_data/' + fname + '_ob_data.npy', allow_pickle=True).item()
test_cont_load = np.load(data_path + 'test_data/' + fname + '_cont_data.npy', allow_pickle=True).item()
deets_load = np.load(data_path + 'test_data/' + fname + '_params.npy', allow_pickle=True).item()

#%%
test_ob_all = test_ob_load['test_ob_all']
test_cont_all = test_cont_load['test_cont_all']

params_all = deets_load['params_all']
params_test = deets_load['params_test']
net_idx = deets_load['net_idx']
#rnn_leg = deets_load['rnn_leg']

ob_data = deets_load['ob_data']
cont_data = deets_load['cont_data']
red_dd_seq = deets_load['ob_data']['red_dd_seq']
flist_all = deets_load['flist_all']

trial_len = round((params_test['stim_duration'] + params_test['isi_duration'])/params_test['dt'])
num_net = len(test_ob_all)

#%%
if 0:
    f_data_quality_check(test_ob_all, data_tag = 'oddball')
    
    f_data_quality_check(test_cont_all, data_tag = 'control')

#%%

# def f_plot_multiloss(data_in, params, data_tag = '', sm_bin = 100, bad_rnn = []):
#     for n_gr in range(len(data_in)):
#         plt.figure()
#         leg1 = []
#         for n_rnn in range(len(data_in[n_gr])):
#             colors1 = cm.jet(np.linspace(0,1,len(data_in[n_gr])))
#             if n_rnn not in bad_rnn[n_gr]:
#                 loss1_x, loss1_ob = f_smooth_loss(test_ob_all[n_gr][n_rnn]['lossT'], sm_bin = sm_bin)
#                 plt.semilogy(loss1_x, np.mean(loss1_ob, axis=1), color=colors1[n_rnn])
#                 leg1.append('RNN%d, n=%d, mean_loss=%.1e' % (n_rnn, params[n_gr][n_rnn]['hidden_size'], np.mean(loss1_ob[-1000:,:])))
#         plt.xlabel('time steps [50ms]')   
#         plt.ylabel('log loss')
#         plt.legend(leg1)
#         plt.title('%s; loss ob; sm=%d' % (rnn_leg[n_gr], sm_bin))
        
if 0:
    # plt.close('all')
    sm_bin = 100
    
    for n_gr in range(1):
        plt.figure()
        leg1 = []
        for n_rnn in range(len(test_cont_all[n_gr])):
            colors1 = cm.jet(np.linspace(0,1,len(test_cont_all[n_gr])))
            if n_rnn not in bad_rnn[n_gr]:
                loss1_x, loss1_ob = f_smooth_loss(test_ob_all[n_gr][n_rnn]['lossT'], sm_bin = sm_bin)
                plt.semilogy(loss1_x, np.mean(loss1_ob, axis=1), color=colors1[n_rnn])
                leg1.append('RNN%d, n=%d, mean_loss=%.1e' % (n_rnn, params_all[n_gr][n_rnn]['hidden_size'], np.mean(loss1_ob[-500:,:])))
        plt.xlabel('time steps [50ms]')   
        plt.ylabel('log loss')
        plt.legend(leg1)
        plt.title('%s; loss ob; sm=%d' % (rnn_leg[n_gr], sm_bin))
        
        
    for n_gr in range(1):
        plt.figure()
        leg1 = []
        for n_rnn in range(len(test_cont_all[n_gr])):
            colors1 = cm.jet(np.linspace(0,1,len(test_cont_all[n_gr])))
            if n_rnn not in bad_rnn[n_gr]:
                loss1_x, loss1_ob = f_smooth_loss(test_cont_all[n_gr][n_rnn]['lossT'], sm_bin = sm_bin)
                plt.semilogy(loss1_x, np.mean(loss1_ob, axis=1), color=colors1[n_rnn])
                leg1.append('RNN%d, n=%d, mean_loss=%.1e' % (n_rnn, params_all[n_gr][n_rnn]['hidden_size'], np.mean(loss1_ob[-500:,:])))
        plt.xlabel('time steps [50ms]')   
        plt.ylabel('log loss')
        plt.legend(leg1)
        plt.title('%s; loss control; sm=%d' % (rnn_leg[n_gr], sm_bin))


#%% remove bad rnns

if len(test_ob_all[1]) == len(flist_all[1]):
    for n_gr in range(num_net):
        if len(bad_rnn[n_gr]):

            bad_rnn[n_gr].sort(reverse = True)
            for n_rnn in bad_rnn[n_gr]:
                test_ob_all[n_gr].pop(n_rnn)
                test_cont_all[n_gr].pop(n_rnn)
                print('removing tr%d, rnn%d' % (n_gr, n_rnn))
                if len(flist_all[n_gr]):
                    print(flist_all[n_gr][n_rnn])
                net_idx[n_gr].pop(n_rnn)
            
            
#%%
for n_gr in range(num_net):
    for n_rnn in range(len(test_ob_all[n_gr])):
        
        f_cut_reshape_rates_wrap(test_ob_all[n_gr][n_rnn], params_all[n_gr][n_rnn], num_skip_trials = num_skip_trials)
        f_run_dred_wrap(test_ob_all[n_gr][n_rnn], subtr_mean=dred_subtr_mean, method=dred_met)
        
        f_cut_reshape_rates_wrap(test_cont_all[n_gr][n_rnn], params_all[n_gr][n_rnn], num_skip_trials = num_skip_trials)
        f_run_dred_wrap(test_cont_all[n_gr][n_rnn], subtr_mean=dred_subtr_mean, method=dred_met)


trials_oddball_ctx_cut = ob_data['trials_oddball_ctx'][num_skip_trials:,:]
trials_oddball_freq_cut = ob_data['trials_oddball_freq'][num_skip_trials:,:]

trials_cont_cut = cont_data['trials_control_freq'][num_skip_trials:,:]

net_idx_fl = np.concatenate(net_idx)
# rates3d_in = test_ob_all[n_gr][n_rnn]['rates']
# params = params_all[n_gr][n_rnn]

#%% plot dred variance

plt.figure()
for n_gr in range(num_net):
    for n_rnn in range(len(test_ob_all[n_gr])):
        var1 = test_ob_all[n_gr][n_rnn]['exp_var']
        x_lab = np.linspace(0, 1, len(var1))
        plt.plot(x_lab, var1)

pr_ob = []
pr_cont = []
num_ner = []
for n_gr in range(num_net):
    num_rnn = len(test_ob_all[n_gr])
    pr1 = np.zeros((num_rnn))
    pr2 = np.zeros((num_rnn))
    num_neur1 = np.zeros((num_rnn))
    for n_rnn in range(num_rnn):
        var1 = test_ob_all[n_gr][n_rnn]['exp_var']
        pr1[n_rnn] = np.sum(var1)**2/np.sum(var1**2)
        
        var2 = test_cont_all[n_gr][n_rnn]['exp_var']
        pr2[n_rnn] = np.sum(var2)**2/np.sum(var2**2)
        
        num_neur1[n_rnn] = len(var1)

    pr_ob.append(pr1)
    pr_cont.append(pr2)
    num_ner.append(num_neur1)

colors1=['blue', 'magenta', 'green']


plt.figure()
for n_gr in range(num_net):
    plt.plot(pr_ob[n_gr], num_ner[n_gr], '.', color=colors1[n_gr])
plt.title('participation ration vs neuron number')


col1 = ['blue', 'magenta', 'green', ]
col2 = ['darkblue', 'darkmagenta', 'darkgreen']

do_sem = 1
plt.figure()
plt.bar(['Ob trained\nob input', 'Freq trained\nob input', 'Untrained\nob input', 'Ob trained\ncont input', 'Freq trained\ncont input', 'Untrained\ncont input'], np.zeros(num_net*2))
for n_net in range(num_net):
    y_data = pr_ob[n_net]
    if do_sem:
        stds = np.std(y_data)/np.sqrt(len(y_data)-1)
    else:
        stds = np.std(y_data)
    if len(colors1):
        plt.bar(n_net, np.mean(y_data), color=col1[n_net], alpha=0.5, edgecolor=col1[n_net])
    else:
        plt.bar(n_net, np.mean(y_data))
    plt.plot(n_net, np.mean(y_data), '_', color='black', mew=2, markersize=40)
    plt.errorbar(n_net, np.mean(y_data), stds, fmt='o', color='black', mew=2, markersize=5, linewidth=2, capsize=10)
plt.title('PR oddball')

for n_net in range(num_net):
    n_net2 = n_net+3
    y_data = pr_cont[n_net]
    if do_sem:
        stds = np.std(y_data)/np.sqrt(len(y_data)-1)
    else:
        stds = np.std(y_data)
    if len(colors1):
        plt.bar(n_net2, np.mean(y_data), color=col2[n_net], alpha=0.5, edgecolor=col2[n_net])
    else:
        plt.bar(n_net2, np.mean(y_data))
    plt.plot(n_net2, np.mean(y_data), '_', color='black', mew=2, markersize=40)
    plt.errorbar(n_net2, np.mean(y_data), stds, fmt='o', color='black', mew=2, markersize=5, linewidth=2, capsize=10)
plt.title('Participation ratio')


do_sem = 1
plt_lim = 4
leg_all = []
leg_lab = ['Ob trained ob input', 'Freq trained ob input', 'Untrained ob input', 'Ob trained cont input', 'Freq trained cont input', 'Untrained cont input']
leg_lab = ['Ob trained\nob input', 'Freq trained\nob input', 'Untrained\nob input', 'Ob trained\ncont input', 'Freq trained\ncont input', 'Untrained\ncont input']
plt.figure()
for n_gr in range(num_net):
    num_rnn = len(test_ob_all[n_gr])
    exp_all = np.zeros((num_rnn, 25))
    for n_rnn in range(num_rnn):
        exp_all[n_rnn,:] = test_ob_all[n_gr][n_rnn]['exp_var'][:25]
        
    y_mean = np.mean(exp_all, axis=0)
    if do_sem:
        y_sem = np.std(exp_all, axis=0)/np.sqrt(num_rnn-1)
    else:
        y_sem = np.std(exp_all, axis=0)
        
    x_lab = np.arange(25)+1

    l1 = f_plot_shadederrorbar2(x_lab[:plt_lim], y_mean[:plt_lim], y_sem[:plt_lim], color=col1[n_gr])
    leg_all.append(l1[0])
for n_gr in range(num_net):
    num_rnn = len(test_ob_all[n_gr])
    exp_all = np.zeros((num_rnn, 25))
    for n_rnn in range(num_rnn):
        exp_all[n_rnn,:] = test_cont_all[n_gr][n_rnn]['exp_var'][:25]
        
    y_mean = np.mean(exp_all, axis=0)
    if do_sem:
        y_sem = np.std(exp_all, axis=0)/np.sqrt(num_rnn-1)
    else:
        y_sem = np.std(exp_all, axis=0)
        
    x_lab = np.arange(25)+1

    l1 = f_plot_shadederrorbar2(x_lab[:plt_lim], y_mean[:plt_lim], y_sem[:plt_lim], color=col2[n_gr])
    leg_all.append(l1[0])

plt.legend(leg_all, leg_lab)
plt.title('RNN activity dimensionality')
plt.xlabel('PCA components')
plt.ylabel('Explained variance')

# f_save_fig(plt.figure(4), path=fig_path, name_tag='')


#%% plot mmn population response
# plt.close('all')
base_sub = True
split_pos_neg = False

n_gr = 0

#for n_gr in range(num_net):
for n_rnn in range(len(test_ob_all[n_gr])):
    #f_plot_mmn(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_ctx_cut, params_all[n_gr][n_rnn], title_tag='ob trained RNN')
    f_plot_mmn2(trials_oddball_ctx_cut, test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_cont_cut, test_cont_all[n_gr][n_rnn]['rates4d_cut'], params_all[n_gr][n_rnn], red_dd_seq, title_tag='%s RNN%d' % (rnn_leg[n_gr], n_rnn+1), baseline_subtract=base_sub, split_pos_cells=split_pos_neg)
    #f_plot_traj_speed(test_ob_all[n_rnn]['rates'], ob_data1, n_run, start_idx=1000, title_tag= 'trained RNN %d; run %d' % (n_rnn, n_run))

# rates4d_cut = test_ob_all[n_gr][n_rnn]['rates4d_cut']
# rates_cont_freq4d_cut = test_cont_all[n_gr][n_rnn]['rates4d_cut']
#%% MMN pool
base_sub = True
comb_met = 'networks'  # networks, cells

rdc_all = []

for n_gr in range(num_net):
    rdc_all2 = []
    for n_rnn in range(len(test_ob_all[n_gr])):
        rates4d_cut = test_ob_all[n_gr][n_rnn]['rates4d_cut']
        
        trial_len, num_tr, num_runs, num_cells = rates4d_cut.shape
        
        plot_t1 = f_plot_t(trial_len, params_all[n_gr][n_rnn]['dt'])
        
        rdc_all2.append(f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_cont_cut, test_cont_all[n_gr][n_rnn]['rates4d_cut'], params_all[n_gr][n_rnn], red_dd_seq, baseline_subtract=base_sub))
    rdc_all.append(rdc_all2)


# by all cells
for n_net in range(num_net):  
    rdc_all2 = rdc_all[n_net]
    num_rnns = len(rdc_all2)
    
    rdc_all32 = np.concatenate(rdc_all2, axis=3)
    
    rdc_all21 = []
    for n_rnn in range(num_rnns):
        rdc_all21.append(np.mean(rdc_all2[n_rnn], axis=3)[:,:,:,None])
    rdc_all31 = np.concatenate(rdc_all21, axis=3)
    
    trial_len, _, num_freqs, num_cells = rdc_all32.shape
    
    if comb_met == 'networks':
        rdc_all3 = rdc_all31
    else:
        rdc_all3 = rdc_all32
        comb_met = 'cells'
    
    num_poits = rdc_all3.shape[3]
    
    rdc_all4 = np.reshape(rdc_all3, (trial_len, 3, num_freqs*num_poits), order='F')

    mmn_mean = np.mean(rdc_all4, axis=2)
    mmn_sem = np.std(rdc_all4, axis=2)/np.sqrt(num_freqs*num_poits-1)
    
    colors_ctx = ['blue', 'red', 'black']
        
    if num_cells:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for n_ctx in range(3):
            ax.plot(plot_t1, mmn_mean[:,n_ctx], color=colors_ctx[n_ctx], zorder=2*n_ctx+1)
            ax.fill_between(plot_t1, mmn_mean[:,n_ctx]-mmn_sem[:,n_ctx], mmn_mean[:,n_ctx]+mmn_sem[:,n_ctx], color=colors_ctx[n_ctx], alpha=0.2, zorder=2*n_ctx)
        ylim = ax.get_ylim()
        ax.add_patch(Rectangle([0, ylim[0]], 0.5, ylim[1]-ylim[0], color='blue', alpha=0.2, zorder=0))
        ax.set_xlim([-0.2, 0.75])
        plt.title('%s RNN population trial ave by %s; %d rnns, %d cells' % (rnn_leg[n_net], comb_met, num_rnns, num_cells))
        plt.xlabel('Time (sec)')
        plt.ylabel('Response')
        
# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')
# f_save_fig(plt.figure(3), path=fig_path, name_tag='')


#%% compute MMN response magnitude

base_sub = True
on_period = [0.2, 0.5]

dr_ratio = []
mmn_mag = []
mmn_magn = []

for n_gr in range(num_net):
    for n_rnn in range(len(test_ob_all[n_gr])):
        trial_ave_rdc = f_get_rdc_trav(trials_oddball_ctx_cut, test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_cont_cut, test_cont_all[n_gr][n_rnn]['rates4d_cut'], params_all[n_gr][n_rnn], red_dd_seq, baseline_subtract=base_sub)
        
        plot_t1 = f_plot_t(trial_len, params_all[n_gr][n_rnn]['dt'])
        on_time = (plot_t1>=on_period[0])*(plot_t1<=on_period[1])
        
        rdc_mag = np.mean(np.mean(np.mean(trial_ave_rdc[on_time,:,:,:], axis=0), axis=2), axis=1)
        dr_ratio.append(np.abs(rdc_mag[1]/rdc_mag[0]))
        mmn_mag.append((rdc_mag[1] - rdc_mag[0]))
        mmn_magn.append((rdc_mag[1] - rdc_mag[0])/np.max(trial_ave_rdc))


dr_ratio = np.array(dr_ratio)
mmn_mag = np.array(mmn_mag)
mmn_magn = np.array(mmn_magn)

#%% context decoding
# trial_types = [1, 0]

plot_t1 = f_plot_t(trial_len, params_all[0][0]['dt'])
trial_stim_on = f_get_stim_on_bins(trial_len, params_all[0][0])
    
stim_loc = trials_oddball_ctx_cut

rates_in = []
shuff_stim_type = []
shuff_bins = []
leg_all = []
for n_gr in range(num_net):
    for n_rnn in range(len(test_ob_all[n_gr])):
        rates_in.append(test_ob_all[n_gr][n_rnn]['rates4d_cut'])
        shuff_stim_type.append(0)
        shuff_bins.append(0)
        leg_all.append('%s, rnn%d' % (rnn_leg[n_gr], n_rnn))
        #params_all[n_gr][n_rnn]
        

# rates_in = [test_oddball_ctx['rates4d_cut'], testf_oddball_ctx['rates4d_cut'], test0_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut'], test_oddball_ctx['rates4d_cut']]
# leg_all = ['ob trained', 'freq trained', 'untrained', 'ob stim shuff', 'ob bin shuff']
# shuff_stim_type = [0, 0, 0, 1, 0]
# shuff_bins = [0, 0, 0, 0, 1]

x_data, y_data = f_sample_trial_data_dec(rates_in, stim_loc, [1, 0])

perform1_final, perform1_binwise, perform1_y_is_cat = f_run_one_shot_dec(x_data, y_data, trial_stim_on, shuff_stim_type, shuff_bins, stim_on_train=False, num_cv=5, equalize_y_input=True)

f_plot_one_shot_dec_bycat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, leg_all, ['deviant', 'redundant'], ['pink', 'lightblue'])

f_plot_one_shot_dec_bycat2(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, net_idx_fl, rnn_leg, ['deviant', 'redundant'], ['pink', 'lightblue'])

# f_save_fig(plt.figure(3), path=fig_path, name_tag='')
# f_save_fig(plt.figure(4), path=fig_path, name_tag='')
#%%
# plt.close('all')
plt.figure()
for n_net in range(num_net):
    plt.plot(dr_ratio[net_idx_fl==n_net], perform1_final[net_idx_fl==n_net], '.')
plt.title('D/R ratio vs ob performance')
plt.xlabel('dev/red ratio')
plt.ylabel('Oddball performance')
plt.legend(rnn_leg)

plt.figure()
for n_net in range(num_net):
    plt.plot(mmn_mag[net_idx_fl==n_net], perform1_final[net_idx_fl==n_net], '.')
plt.title('MMN vs ob performance')
plt.xlabel('MMN magnitude')
plt.ylabel('Oddball performance')
plt.legend(rnn_leg)

plt.figure()
for n_net in range(num_net):
    plt.plot(mmn_magn[net_idx_fl==n_net], perform1_final[net_idx_fl==n_net], '.')
plt.title('Normalized MMN vs ob performance')
plt.xlabel('Normalized MMN')
plt.ylabel('Oddball performance')
plt.legend(rnn_leg)

# f_save_fig(plt.figure(4), path=fig_path, name_tag='')

#%% decoding frequency identity from control inputs

colors10 = cm.jet(np.linspace(0,1, len(cont_data['control_stim'])))
colors10[:,3] = 0.3

plot_t1 = f_plot_t(trial_len, params_all[0][0]['dt'])
trial_stim_on = f_get_stim_on_bins(trial_len, params_all[0][0])
    
stim_loc = trials_cont_cut

rates_in = []
shuff_stim_type = []
shuff_bins = []
leg_all = []
for n_gr in range(num_net):
    for n_rnn in range(len(test_cont_all[n_gr])):
        rates_in.append(test_cont_all[n_gr][n_rnn]['rates4d_cut'])
        shuff_stim_type.append(0)
        shuff_bins.append(0)
        leg_all.append('%s, rnn%d' % (rnn_leg[n_gr], n_rnn))
        #params_all[n_gr][n_rnn]


x_data, y_data = f_sample_trial_data_dec(rates_in, stim_loc, cont_data['control_stim']) #  np.hstack((cont_data['control_stim'],[0]))

perform1_final, perform1_binwise, perform1_y_is_cat = f_run_one_shot_dec(x_data, y_data, trial_stim_on, shuff_stim_type, shuff_bins, stim_on_train=False, num_cv=5, equalize_y_input=True)

f_plot_one_shot_dec_avecat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, net_idx_fl, rnn_leg)

#f_plot_one_shot_dec_bycat(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, leg_all, [], colors10)

#f_plot_one_shot_dec_bycat2(perform1_final, perform1_binwise, perform1_y_is_cat, plot_t1, net_idx_fl, rnn_leg, ['deviant', 'redundant'], ['pink', 'lightblue'])

#%%

# if 'red_count' not in ob_data1.keys():
#     ctx_data = ob_data1['trials_oddball_ctx']
#     num_trials, num_runs = ctx_data.shape
#     red_count = np.zeros((num_trials, num_runs))
#     for n_run in range(num_runs):
#         red_count1 = 1
#         for n_gr in range(num_trials):
#             if not ctx_data[n_gr,n_run]:
#                 red_count[n_gr,n_run] = red_count1
#                 red_count1 += 1
#             else:
#                 red_count1 = 1
        
#     ob_data1['red_count'] = red_count

#%%
colors1 = cm.jet(np.linspace(0,1,params_test['num_freq_stim']))
if 0:
    plt.figure()
    plt.imshow(colors1[:,:3].reshape((50,1,3)), aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])
    
#%%get tau from full traces
# plt.close('all')

# need to clip initial 50 trials after oddball starts
trials_start = params_test['num_prepend_zeros'] + 50 - num_skip_trials

tau_net_all = []
tau_cell_all = []
tau_cell_runmean_all = []

for n_gr in range(num_net):
    tau_net_all2 = []
    tau_cell_all2 = []
    tau_cell_runmean_all2 = []
    
    for n_rnn in range(len(test_ob_all[n_gr])):
        print('gr %d; rnn %d' %(n_gr, n_rnn))

        rates4d = test_ob_all[n_gr][n_rnn]['rates4d_cut'][:,trials_start:,:,:]
        trial_len, num_tr, num_runs, num_cells = rates4d.shape
        rates3d = np.reshape(rates4d, [trial_len*num_tr, num_runs, num_cells], order='F')

        tau_net = np.zeros(num_runs)
        tau_cell = np.zeros((num_runs, num_cells))
        tau_cell_runmean = np.zeros((num_cells))
        has_data_neur = np.zeros((num_runs, num_cells), dtype=bool)
        
        
        #temp_ave4d, trial_data_sort, num_dd_trials = f_trial_ave_ctx_pad2(rates4d, trials_oddball_ctx_cut, pre_dd = 1, post_dd = 16, limit_1_dd=False, max_trials=999, shuffle_trials=False)
        #temp_ave3d = np.reshape(temp_ave4d, (trial_len*temp_ave4d.shape[1],num_runs,num_cells), order='F')

        for n_run in range(num_runs):
            
            #plt.figure(); plt.plot(np.reshape(rates4d[:,:,n_run,:], (trial_len*num_tr, num_cells)))
            rates_run = rates3d[:,n_run,:]
            base_vec = np.mean(rates_run, axis=0)
            tr_dist = cdist(np.reshape(base_vec, (1,num_cells)), rates_run, 'euclidean')[0]
            tau_net1, _ = f_get_trace_tau(tr_dist, sm_bin = 0)
            tau_net[n_run] = tau_net1*params_all[n_gr][n_rnn]['dt']
            
            
            for n_nr in range(num_cells):
                #neur = np.reshape(temp_ave4d[:,:,n_run,n_nr], (trial_len*temp_ave4d.shape[1]))
                
                neur = rates3d[:,n_run,n_nr]
                # plt.figure(); plt.plot(neur)
                
                if np.sum(neur) > 0.1:
                    
                    tau_neur1, _ = f_get_trace_tau(neur, sm_bin = 0)
                        
                    tau_cell[n_run, n_nr] = tau_neur1*params_test['dt']
                    has_data_neur[n_run, n_nr] = True
        
        for n_nr in range(num_cells):
            if np.sum(has_data_neur[:,n_nr]):
                tau_cell_runmean[n_nr] = np.mean(tau_cell[has_data_neur[:,n_nr],n_nr])
            
        tau_net_all2.append(tau_net)
        tau_cell_all2.append(tau_cell[has_data_neur])
        tau_cell_runmean_all2.append(tau_cell_runmean[np.sum(has_data_neur, axis=0).astype(bool)])
        
    tau_net_all.append(tau_net_all2)
    tau_cell_all.append(tau_cell_all2)
    tau_cell_runmean_all.append(tau_cell_runmean_all2)

tau_net_all_fl = []
tau_cell_all_fl = []
tau_cell_runmean_all_fl = []
for n_net in range(num_net):
    tau_net_all_fl.append(np.concatenate(tau_net_all[n_net]))
    tau_cell_all_fl.append(np.concatenate(tau_cell_all[n_net]))
    tau_cell_runmean_all_fl.append(np.concatenate(tau_cell_runmean_all[n_net]))

#%%
f_plot_cat_data2(tau_net_all, rnn_leg, title_tag = 'Tau network', do_log=False)
plt.ylabel('tau (sec)');
f_plot_cat_data2(tau_cell_all, rnn_leg, title_tag = 'Tau neurons', do_log=False)
plt.ylabel('tau (sec)');
f_plot_cat_data2(tau_cell_runmean_all, rnn_leg, title_tag = 'Tau neurons run mean', do_log=False)
plt.ylabel('tau (sec)');

f_plot_cat_data_violin(tau_net_all_fl, rnn_leg, title_tag = 'Tau network', points=1000, mean_std=True, showmedians=True, showmeans=False, quantile = [0.05, 0.95], colors=['blue', 'magenta', 'green'], do_log=True)
plt.ylabel('tau (sec)');
f_plot_cat_data_violin(tau_cell_all_fl, rnn_leg, title_tag = 'Tau neurons', points=1000, mean_std=True, showmedians=True, showmeans=False, quantile = [0.05, 0.95], colors=['blue', 'magenta', 'green'], do_log=True)
plt.ylabel('tau (sec)');
f_plot_cat_data_violin(tau_cell_runmean_all_fl, rnn_leg, title_tag = 'Tau neurons run mean', points=1000, mean_std=True, showmedians=True, showmeans=False, quantile = [0.05, 0.95], colors=['blue', 'magenta', 'green'], do_log=True)
plt.ylabel('tau (sec)');

f_plot_cat_data_bar(tau_net_all_fl, rnn_leg, title_tag = 'Tau network', colors=['blue', 'magenta', 'green'])
plt.ylabel('tau (sec)');
f_plot_cat_data_bar(tau_cell_all_fl, rnn_leg, title_tag = 'Tau neurons', colors=['blue', 'magenta', 'green'])
plt.ylabel('tau (sec)');
f_plot_cat_data_bar(tau_cell_runmean_all_fl, rnn_leg, title_tag = 'Tau neurons run mean', colors=['blue', 'magenta', 'green'])
plt.ylabel('tau (sec)');

# y_data_in = tau_net_all_fl

# f_save_fig(plt.figure(14), path=fig_path, name_tag='scatter')
# f_save_fig(plt.figure(15), path=fig_path, name_tag='scatter')
# f_save_fig(plt.figure(16), path=fig_path, name_tag='scatter')
# f_save_fig(plt.figure(4), path=fig_path, name_tag='violin')
# f_save_fig(plt.figure(5), path=fig_path, name_tag='violin')
# f_save_fig(plt.figure(6), path=fig_path, name_tag='violin')
# f_save_fig(plt.figure(7), path=fig_path, name_tag='bar')
# f_save_fig(plt.figure(8), path=fig_path, name_tag='bar')
# f_save_fig(plt.figure(9), path=fig_path, name_tag='bar')

# tau_net_all_fl_log = []
# tau_cell_all_fl_log = []
# for n_net in range(num_net):
#     tau_net_all_fl_log.append(np.log(np.concatenate(tau_net_all[n_net])))
#     tau_cell_all_fl_log.append(np.log(np.concatenate(tau_cell_all[n_net])))

# f_plot_cat_data_violin(tau_net_all_fl_log, rnn_leg, title_tag = 'Tau network', points=100, mean_std=False, showmedians=True, showmeans=False, quantile = [0.05, 0.95], colors=['blue', 'magenta', 'green'], do_log=False)
# plt.ylabel('tau (sec)');
# f_plot_cat_data_violin(tau_cell_all_fl_log, rnn_leg, title_tag = 'Tau neurons', points=100, mean_std=False, showmedians=True, showmeans=False, quantile = [0.05, 0.95], colors=['blue', 'magenta', 'green'], do_log=False)
# plt.ylabel('tau (sec)');

    
#%% plot control vectors mags of indiv vs trial ave
red_tr_idx = -3
base_time1 = [-.250, 0]
on_time1 = [.2, .5]

trials_cont_all = []
trials_cont2d_all = []
trials_dev_all = []
trials_red_all = []
for n_gr in range(num_net):
    trials_cont_all2 = []
    trials_cont2d_all2 = []
    trials_dev_all2 = []
    trials_red_all2 = []
    for n_rnn in range(len(test_cont_all[n_gr])):
        trials_cont = f_gather_cont_trials(test_cont_all[n_gr][n_rnn]['rates4d_cut'], trials_cont_cut, red_dd_seq)
        trials_cont_vec = f_analyze_cont_trial_vectors(trials_cont, params_all[n_gr][n_rnn], base_time = base_time1, on_time = on_time1)
        trials_cont_all2.append(trials_cont_vec)
        
        # 2d array including last trial history (last, current)
        trials_cont2d = f_gather_cont_trials2d(test_cont_all[n_gr][n_rnn]['rates4d_cut'], trials_cont_cut, red_dd_seq)
        trials_cont_vec2d = f_analyze_rd_trial_vectors(trials_cont2d, params_all[n_gr][n_rnn], base_time = base_time1, on_time = on_time1)
        trials_cont2d_all2.append(trials_cont_vec2d)
        
        trials_rd_dev = f_gather_dev_trials(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_ctx_cut, red_dd_seq)
        trials_dev_vec = f_analyze_rd_trial_vectors(trials_rd_dev, params_all[n_gr][n_rnn], base_time = base_time1, on_time = on_time1)
        trials_dev_all2.append(trials_dev_vec)
        
        trials_rd_red = f_gather_red_trials(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_freq_cut, trials_oddball_ctx_cut, red_dd_seq, red_idx = red_tr_idx)
        trials_red_vec = f_analyze_rd_trial_vectors(trials_rd_red, params_all[n_gr][n_rnn], base_time = base_time1, on_time = on_time1)
        trials_red_all2.append(trials_red_vec)
        
    trials_cont_all.append(trials_cont_all2)
    trials_cont2d_all.append(trials_cont2d_all2)
    trials_dev_all.append(trials_dev_all2)
    trials_red_all.append(trials_red_all2)

# trials_rd = trials_rd_dev
# params = params_all[n_gr][n_rnn]


#%% trial to trial cosine similarity plots

# rates_cont4d = test_cont_all[0][0]['rates4d_cut']
# trials_cont_cut = trials_cont_cut
# freqs_list = red_dd_seq


in_type = ['deviant', 'control', 'redundant']
base_sub = 1
clim1 = [0, 1]

n_gr = 1

for n_in in range(len(in_type)):
    cs_all = []
    
    for n_rnn in range(len(test_cont_all[n_gr])):
        
        if in_type[n_in] == 'deviant':
            trials1 = f_gather_dev_trials(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_ctx_cut, red_dd_seq)
        elif in_type[n_in] == 'control':
            trials1 = f_gather_cont_trials2d(test_cont_all[n_gr][n_rnn]['rates4d_cut'], trials_cont_cut, red_dd_seq)
        elif in_type[n_in] == 'redundant':
            trials1 = f_gather_red_trials(test_ob_all[n_gr][n_rnn]['rates4d_cut'], trials_oddball_freq_cut, trials_oddball_ctx_cut, red_dd_seq)
        #trials1 = f_gather_cont_trials2d(test_cont_all[n_gr][n_rnn]['rates4d_cut'], trials_cont_cut, red_dd_seq)
        
        num_red = len(trials1)
        num_dev = len(trials1[0])
        
        trials_all1 = []
        
        for n_dev in range(num_dev):
            for n_red in range(num_red):
                trial_data1 = np.concatenate(trials1[n_red][n_dev], axis=1)
                trial_data2 = np.mean(trial_data1, axis=1)[:,None,:]
                trials_all1.append(trial_data2)
        
        trials_all2 = np.concatenate(trials_all1, axis=1)
        trials_all3 = np.mean(trials_all2[10:15,:,:], axis=0)
        
        if base_sub:
            trials_all3 = trials_all3 - np.mean(trials_all2[:5,:,:], axis=0) 

        cs1 = 1 - squareform(pdist(trials_all3, 'cosine'))
        
        cs_all.append(cs1[:,:,None])
        
        
    cs3 = np.mean(np.concatenate(cs_all, axis=2), axis=2)
    
    plt.figure()
    plt.imshow(cs3)
    plt.colorbar()
    plt.clim(clim1)
    plt.title('%s; %s' % (rnn_leg[n_gr], in_type[n_in]))
    plt.xlabel('sorted current, last trial')
    plt.ylabel('sorted current, last trial')


# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')
# f_save_fig(plt.figure(3), path=fig_path, name_tag='')

# plot axis colors
if 0:
    colors2 = cm.jet(np.linspace(0,1,num_red))
    
    col_im = colors2[:,:3].reshape((num_red,1,3))
    
    col_im2 = np.tile(col_im, [20, 1, 1])

    plt.figure()
    plt.imshow(col_im, aspect=.2)
    plt.ylabel('color map')
    plt.xticks([])
    
    plt.figure()
    plt.imshow(col_im2, aspect='auto')
    plt.ylabel('color map')
    plt.xticks([])
    
#%%
# plt.close('all')

n_gr = 0

keys_plot = [
            'base_dist_mean',
            'on_dist_mean',
            'indiv_trial_mag_mean',
            'mean_vec_mag',
            'indiv_trial_angles_mean',
            ]

for key1 in keys_plot:
    plt.figure()
    for n_rnn in range(len(trials_cont_all[n_gr])):    
        plt.plot(trials_cont_all[n_gr][n_rnn][key1])
    plt.title('%s gr%d' % (key1, n_gr))


n_rnn = 1

f_plot_cont_vec_data(trials_cont_all[0][n_rnn], red_dd_seq)

f_plot_rd_vec_data(trials_dev_all[0][n_rnn], ctx_tag = 'deviant')

f_plot_rd_vec_data(trials_red_all[0][n_rnn], ctx_tag = 'redundant %d' % red_tr_idx)

f_plot_ctx_vec_data(trials_cont_all[0][n_rnn], trials_dev_all[0][n_rnn], trials_red_all[0][n_rnn])


#%% plot vector magnitudes vs context

# dict_keys(['base_dist_mean', 'base_dist_std', 'on_dist_mean', 'on_dist_std', 'indiv_mag_mean', 'indiv_mag_std', 'mean_mag', 'angles_mean', 'angles_std', 'mean_vec'])


n_gr = 0

temp_dev_mags = []
temp_red_mags = []
for n_rnn in range(len(trials_cont_all[n_gr])):  
    temp_dev_mags.append(trials_dev_all[n_gr][n_rnn]['indiv_trial_mag_mean'][:,:,None])
    temp_red_mags.append(trials_red_all[n_gr][n_rnn]['indiv_trial_mag_mean'][:,:,None])

temp_dev_mags2 = np.mean(np.concatenate(temp_dev_mags, axis=2), axis=2)
temp_red_mags2 = np.mean(np.concatenate(temp_red_mags, axis=2), axis=2)


plt.figure()
plt.imshow(temp_dev_mags2, cmap=cmap1)
plt.colorbar()
plt.title('%s; trial ave mag deviants' % rnn_leg[n_gr])
plt.ylabel('deviant freq')
plt.xlabel('redundant freq')

plt.figure()
plt.imshow(temp_red_mags2, cmap=cmap1)
plt.colorbar()
plt.title('%s; trial ave mag redundants' % rnn_leg[n_gr])
plt.ylabel('deviant freq')
plt.xlabel('redundant freq')

plt.figure()
plt.imshow(temp_dev_mags2/temp_red_mags2, cmap=cmap1)
plt.colorbar()
plt.title('%s; trial ave mag dev-red_ratio' % rnn_leg[n_gr])
plt.ylabel('deviant freq')
plt.xlabel('redundant freq')

plt.figure()
plt.imshow(temp_dev_mags2-temp_red_mags2, cmap=cmap1)
plt.colorbar()
plt.title('%s; trial ave mag dev-red diff' % rnn_leg[n_gr])
plt.ylabel('deviant freq')
plt.xlabel('redundant freq')
    
# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')
# f_save_fig(plt.figure(3), path=fig_path, name_tag='')
# f_save_fig(plt.figure(4), path=fig_path, name_tag='')

x_range, y_diff_means, y_diff_stds = f_get_diags_data(temp_dev_mags2-temp_red_mags2)
_, y_dev_means, y_dev_stds = f_get_diags_data(temp_dev_mags2)
_, y_red_means, y_red_stds = f_get_diags_data(temp_red_mags2)
_, y_rat_means, y_rat_stds = f_get_diags_data(temp_dev_mags2/temp_red_mags2)

plt.figure()
l1 = f_plot_shadederrorbar2(x_range, y_dev_means, y_dev_stds, color='red')
l2 = f_plot_shadederrorbar2(x_range, y_red_means, y_red_stds, color='blue')
plt.title('%s; context response' % rnn_leg[n_gr])
plt.xlabel('Frequency difference')
plt.ylabel('Response magnitude')
plt.legend(l1 + l2, ['deviant', 'redundant'])


plt.figure()
f_plot_shadederrorbar2(x_range, y_diff_means, y_diff_stds, color='gray')
plt.title('%s; dev minus red' % rnn_leg[n_gr])
plt.xlabel('Frequency difference')
plt.ylabel('Response diff magnitude')

plt.figure()
f_plot_shadederrorbar2(x_range, y_rat_means, y_rat_stds, color='gray')
plt.title('%s; dev-red ratio' % rnn_leg[n_gr])
plt.xlabel('Frequency difference')
plt.ylabel('Response ratio magnitude')



y_diff_means_all = []
y_diff_stds_all = []
y_rat_means_all = []
y_rat_stds_all = []

for n_gr in range(3):
    
    temp_dev_mags = []
    temp_red_mags = []
    for n_rnn in range(len(trials_cont_all[n_gr])):  
        temp_dev_mags.append(trials_dev_all[n_gr][n_rnn]['indiv_trial_mag_mean'][:,:,None])
        temp_red_mags.append(trials_red_all[n_gr][n_rnn]['indiv_trial_mag_mean'][:,:,None])

    temp_dev_mags2 = np.mean(np.concatenate(temp_dev_mags, axis=2), axis=2)
    temp_red_mags2 = np.mean(np.concatenate(temp_red_mags, axis=2), axis=2)
    
    _, y_diff_means, y_diff_stds = f_get_diags_data(temp_dev_mags2-temp_red_mags2)
    _, y_rat_means, y_rat_stds = f_get_diags_data(temp_dev_mags2/temp_red_mags2)
    
    y_diff_means_all.append(y_diff_means)
    y_diff_stds_all.append(y_diff_stds)
    y_rat_means_all.append(y_diff_means)
    y_rat_stds_all.append(y_diff_stds)


lin_all = []
plt.figure()
for n_gr in range(3):
    l1 = f_plot_shadederrorbar2(x_range, y_diff_means_all[n_gr], y_diff_stds_all[n_gr], color=rnn_color[n_gr])
    lin_all.append(l1[0])
plt.title('%s; dev-red difference' % rnn_leg[n_gr])
plt.xlabel('Frequency difference')
plt.ylabel('Dev minus red')
plt.legend(lin_all, rnn_leg)

lin_all = []
plt.figure()
for n_gr in range(3):
    l1 = f_plot_shadederrorbar2(x_range, y_rat_means_all[n_gr], y_rat_stds_all[n_gr], color=rnn_color[n_gr])
    lin_all.append(l1[0])
plt.title('%s; dev-red ratio' % rnn_leg[n_gr])
plt.xlabel('Frequency difference')
plt.ylabel('Dev-red ratio')
plt.legend(lin_all, rnn_leg)

# f_save_fig(plt.figure(5), path=fig_path, name_tag='')

#%% compare response angles of dev/red, while holding one and varying other
# plt.close('all')

n_gr = 0

cs_dhd_all = []
cs_dhr_all = []
cs_rhd_all = []
cs_rhr_all = []
for n_rnn in range(len(trials_dev_all[n_gr])):  
    
    cos_sim_dev_hold_dev = f_get_cs_same_dev(trials_dev_all[n_gr][n_rnn]['mean_vec'])
    cs_dhd_all.append(cos_sim_dev_hold_dev[:,:,None])
    
    cos_sim_dev_hold_red = f_get_cs_same_red(trials_dev_all[n_gr][n_rnn]['mean_vec'])
    cs_dhr_all.append(cos_sim_dev_hold_red[:,:,None])
    
    cos_sim_red_hold_dev = f_get_cs_same_dev(trials_red_all[n_gr][n_rnn]['mean_vec'])
    cs_rhd_all.append(cos_sim_red_hold_dev[:,:,None])
    
    cos_sim_red_hold_red = f_get_cs_same_red(trials_red_all[n_gr][n_rnn]['mean_vec'])
    cs_rhr_all.append(cos_sim_red_hold_red[:,:,None])
    

cs_dhd_all2 = np.mean(np.concatenate(cs_dhd_all, axis=2), axis=2)
cs_dhr_all2 = np.mean(np.concatenate(cs_dhr_all, axis=2), axis=2)
cs_rhd_all2 = np.mean(np.concatenate(cs_rhd_all, axis=2), axis=2)
cs_rhr_all2 = np.mean(np.concatenate(cs_rhr_all, axis=2), axis=2)

cmin = np.min([cs_dhd_all2, cs_dhr_all2, cs_rhd_all2, cs_rhr_all2])


plt.figure()
plt.imshow(cs_dhd_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim dev resp hold dev' % (rnn_leg[n_gr]))
plt.xlabel('redundant frequency')
plt.ylabel('redundant frequency')

plt.figure()
plt.imshow(cs_dhr_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim dev resp hold red' % (rnn_leg[n_gr]))
plt.xlabel('deviant frequency')
plt.ylabel('deviant frequency')

plt.figure()
plt.imshow(cs_rhd_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim red resp hold dev' % (rnn_leg[n_gr]))
plt.xlabel('redundant frequency')
plt.ylabel('redundant frequency')

plt.figure()
plt.imshow(cs_rhr_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim red resp hold red' % (rnn_leg[n_gr]))
plt.xlabel('deviant frequency')
plt.ylabel('deviant frequency')

# f_save_fig(plt.figure(4), path=fig_path, name_tag='')


#%% compare angles to reference vector
# plt.close('all')

n_gr = 0

cs_c2d_hd_all = []
cs_c2d_hr_all = []
cs_c2r_hd_all = []
cs_c2r_hr_all = []

for n_rnn in range(len(trials_cont_all[n_gr])):  
    
    cs_c2d_hd = f_get_cs_cont_to_ref_same_dev(trials_dev_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
    cs_c2d_hd_all.append(cs_c2d_hd[:,:,None])
        
    cs_c2d_hr = f_get_cs_cont_to_ref_same_red(trials_dev_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
    cs_c2d_hr_all.append(cs_c2d_hr[:,:,None])
    
    cs_c2r_hd = f_get_cs_cont_to_ref_same_dev(trials_red_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
    cs_c2r_hd_all.append(cs_c2r_hd[:,:,None])
    
    cs_c2r_hr = f_get_cs_cont_to_ref_same_red(trials_red_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
    cs_c2r_hr_all.append(cs_c2r_hr[:,:,None])
    

cs_c2d_hd_all2 = np.mean(np.concatenate(cs_c2d_hd_all, axis=2), axis=2)
cs_c2d_hr_all2 = np.mean(np.concatenate(cs_c2d_hr_all, axis=2), axis=2)
cs_c2r_hd_all2 = np.mean(np.concatenate(cs_c2r_hd_all, axis=2), axis=2)
cs_c2r_hr_all2 = np.mean(np.concatenate(cs_c2r_hr_all, axis=2), axis=2)

cmin = np.min([cs_c2d_hd_all2, cs_c2d_hr_all2, cs_c2r_hd_all2, cs_c2r_hr_all2])

plt.figure()
plt.imshow(cs_c2d_hd_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim cont to dev, hold dev' % (rnn_leg[n_gr]))
plt.xlabel('control frequency')
plt.ylabel('redundant frequency')

plt.figure()
plt.imshow(cs_c2d_hr_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim cont to dev, hold red' % (rnn_leg[n_gr]))
plt.xlabel('control frequency')
plt.ylabel('deviant frequency')

plt.figure()
plt.imshow(cs_c2r_hd_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim cont to red, hold dev' % (rnn_leg[n_gr]))
plt.xlabel('control frequency')
plt.ylabel('redundant frequency')

plt.figure()
plt.imshow(cs_c2r_hr_all2, cmap=cmap1)
plt.colorbar()
plt.clim([cmin, 1])
plt.title('%s; cosine sim cont to red, hold red' % (rnn_leg[n_gr]))
plt.xlabel('control frequency')
plt.ylabel('deviant frequency')


# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')
# f_save_fig(plt.figure(3), path=fig_path, name_tag='')
# f_save_fig(plt.figure(4), path=fig_path, name_tag='')

#%% try comparing control to the same deviant ori
# plt.close('all')

clims = [0.4, 1]

n_gr = 0

# compare the mean deviant response to the same freq deviant with variable redundant
dist_all = []
for n_rnn in range(len(trials_cont_all[n_gr])): 
    data_rd = trials_dev_all[n_gr][n_rnn]['mean_vec']
    mean_dev = np.mean(data_rd, axis=0)
    num_red, num_dev, _ = data_rd.shape
    dist_dd = np.zeros((num_dev, num_red))
    for n_freq in range(num_dev):
        
        dist1 = cdist(data_rd[:,n_freq,:], mean_dev[n_freq,:][None,:], metric='cosine').flatten()
        dist_dd[:,n_freq] = 1 - dist1
        
    dist_all.append(dist_dd[:,:,None])
    
dist_all2 = np.mean(np.concatenate(dist_all, axis=2), axis=2)
    
plt.figure()
plt.imshow(dist_all2)
plt.colorbar()
plt.clim(clims)
plt.ylabel('redundant frequency')
plt.xlabel('deviant-deviant frequency')
plt.title('%s; similarity deviant by freq to mean deviant' % rnn_leg[n_gr])

# compare the mean control response to the same freq deviant with variable redundant
dist_all = []
for n_rnn in range(len(trials_cont_all[n_gr])): 
    data_rd = trials_dev_all[n_gr][n_rnn]['mean_vec']
    data_cont = trials_cont_all[n_gr][n_rnn]['mean_vec']
    num_red, num_dev, _ = data_rd.shape
    dist_dd = np.zeros((num_dev, num_red))
    for n_freq in range(num_dev):
        dist1 = cdist(data_rd[:,n_freq,:], data_cont[n_freq,:][None,:], metric='cosine').flatten()
        dist_dd[:,n_freq] = 1 - dist1
        
    dist_all.append(dist_dd[:,:,None])
    
dist_all2 = np.mean(np.concatenate(dist_all, axis=2), axis=2)
    
plt.figure()
plt.imshow(dist_all2)
plt.colorbar()
plt.clim(clims)
plt.ylabel('redundant frequency')
plt.xlabel('deviant-control frequency')
plt.title('%s; similarity deviant by freq to mean control' % rnn_leg[n_gr])


# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')
# f_save_fig(plt.figure(3), path=fig_path, name_tag='')
# f_save_fig(plt.figure(4), path=fig_path, name_tag='')
# f_save_fig(plt.figure(5), path=fig_path, name_tag='')
# f_save_fig(plt.figure(6), path=fig_path, name_tag='')




#%%
clims = [0.1, 1]

n_gr = 0

# compare mean deviant to all other mean deviants
dist_all = []
for n_rnn in range(len(trials_cont_all[n_gr])): 
    data_rd = trials_dev_all[n_gr][n_rnn]['mean_vec']
    mean_dev = np.mean(data_rd, axis=0)
    
    dist1 = 1 - squareform(pdist(mean_dev, metric='cosine'))
    dist_all.append(dist1[:,:,None])
    
dist_all2 = np.mean(np.concatenate(dist_all, axis=2), axis=2)
    
plt.figure()
plt.imshow(dist_all2)
plt.colorbar()
plt.clim(clims)
plt.ylabel('deviant frequency')
plt.xlabel('deviant frequency')
plt.title('%s; mean deviant tuning' % rnn_leg[n_gr])

# compare mean control to all other mean control
dist_all = []
for n_rnn in range(len(trials_cont_all[n_gr])): 
    data_cont = trials_cont_all[n_gr][n_rnn]['mean_vec']

    dist1 = 1 - squareform(pdist(data_cont, metric='cosine'))
    dist_all.append(dist1[:,:,None])
    
dist_all2 = np.mean(np.concatenate(dist_all, axis=2), axis=2)
    
plt.figure()
plt.imshow(dist_all2)
plt.colorbar()
plt.clim(clims)
plt.ylabel('control frequency')
plt.xlabel('control frequency')
plt.title('%s; mean control tuning' % rnn_leg[n_gr])


# compare mean deviant to all other mean control
dist_all = []
for n_rnn in range(len(trials_cont_all[n_gr])): 
    data_rd = trials_dev_all[n_gr][n_rnn]['mean_vec']
    mean_dev = np.mean(data_rd, axis=0)
    data_cont = trials_cont_all[n_gr][n_rnn]['mean_vec']

    dist1 = 1 - cdist(mean_dev, data_cont, metric='cosine')
    dist_all.append(dist1[:,:,None])
    
    
dist_all2 = np.mean(np.concatenate(dist_all, axis=2), axis=2)
    
plt.figure()
plt.imshow(dist_all2)
plt.colorbar()
plt.clim(clims)
plt.ylabel('deviant frequency')
plt.xlabel('control frequency')
plt.title('%s; mean deviant to control tuning' % rnn_leg[n_gr])



# n_gr = 0

# cs_c2d_hd_all = []
# cs_c2d_hr_all = []
# cs_c2r_hd_all = []
# cs_c2r_hr_all = []

# for n_rnn in range(len(trials_cont_all[n_gr])):  
    
#     cs_c2d_hd = f_get_cs_cont_to_dev_vs_red(trials_dev_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
#     cs_c2d_hd_all.append(cs_c2d_hd[:,:,None])
        
#     cs_c2d_hr = f_get_cs_cont_to_ref_same_red(trials_dev_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
#     cs_c2d_hr_all.append(cs_c2d_hr[:,:,None])
    
#     cs_c2r_hd = f_get_cs_cont_to_ref_same_dev(trials_red_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
#     cs_c2r_hd_all.append(cs_c2r_hd[:,:,None])
    
#     cs_c2r_hr = f_get_cs_cont_to_ref_same_red(trials_red_all[n_gr][n_rnn]['mean_vec'], trials_cont_all[n_gr][n_rnn]['mean_vec'])
#     cs_c2r_hr_all.append(cs_c2r_hr[:,:,None])
    
# cs_c2d_hd_all2 = np.mean(np.concatenate(cs_c2d_hd_all, axis=2), axis=2)

# plt.figure()
# plt.imshow(cs_c2d_hd_all2, cmap=cmap1)
# plt.colorbar()
# #plt.clim([cmin, 1])
# plt.title('%s; cosine sim cont to dev, hold dev' % (rnn_leg[n_gr]))
# plt.xlabel('control frequency')
# plt.ylabel('redundant frequency')

#%%

n_gr = 0

# compare mean deviant to all other mean deviants

traces_dev_all = []
traces_red_all = []
traces_cont_all = []
num_cells_all = []
for n_rnn in range(len(trials_cont_all[n_gr])): 
    data_rd = trials_dev_all[n_gr][n_rnn]['on_mean']
    
    num_red, num_dev, num_cells = data_rd.shape
    
    dist_all2 = []
    for n_red in range(num_red):
        dist1 = squareform(pdist(data_rd[n_red,:,:], metric='euclidean'))
        
        x_range, y_means, y_stds = f_get_diags_data(dist1)
        
        traces_dev_all.append(y_means[:,None])
        num_cells_all.append(num_cells)
        
    data_rd = trials_red_all[n_gr][n_rnn]['on_mean']
    
    for n_dev in range(num_dev):
        dist1 = squareform(pdist(data_rd[:,n_dev,:], metric='euclidean'))
        
        x_range, y_means, y_stds = f_get_diags_data(dist1)
        
        traces_red_all.append(y_means[:,None])

    data_cont = trials_cont_all[n_gr][n_rnn]['mean_vec']
    
    dist1 = squareform(pdist(data_cont, metric='euclidean'))
    x_range, y_means, y_stds = f_get_diags_data(dist1)
    
    traces_cont_all.append(y_means[:,None])
           

traces_dev2 = np.concatenate(traces_dev_all, axis=1)
freq_range, num_dsets = traces_dev2.shape
traces_d_mean = np.mean(traces_dev2, axis=1)
traces_d_std = np.std(traces_dev2, axis=1)/np.sqrt(num_dsets-1)

traces_red2 = np.concatenate(traces_red_all, axis=1)
traces_r_mean = np.mean(traces_red2, axis=1)
traces_r_std = np.std(traces_red2, axis=1)/np.sqrt(num_dsets-1)

traces_cont2 = np.concatenate(traces_cont_all, axis=1)
_, num_cont_dsets = traces_cont2.shape
traces_cont_mean = np.mean(traces_cont2, axis=1)
traces_cont_std = np.std(traces_cont2, axis=1)/np.sqrt(num_cont_dsets-1)


start1 = np.floor(freq_range/2).astype(int)

x_vals =(x_range[start1:] + 1)/len(x_range[start1:])
plt.figure()
ax1 = f_plot_shadederrorbar2(x_vals, traces_d_mean[start1:], traces_d_std[start1:], color='red')
ax2 = f_plot_shadederrorbar2(x_vals, traces_r_mean[start1:], traces_r_std[start1:], color='blue')
ax3 = f_plot_shadederrorbar2(x_vals, traces_cont_mean[start1:], traces_cont_std[start1:], color='k')
plt.xlabel('frequency distance')
plt.ylabel('euclidean distance')
plt.title('%s; Response space size' % rnn_leg[n_gr])
plt.legend(ax1+ax2+ax3, ['deviant', 'redundant', 'control'])


low1 = np.floor(np.min([np.mean(traces_dev2, axis=0), np.mean(traces_red2, axis=0)]))
high1 = np.ceil(np.max([np.mean(traces_dev2, axis=0), np.mean(traces_red2, axis=0)]))

plt.figure()
plt.loglog(np.mean(traces_dev2, axis=0), np.mean(traces_red2, axis=0), '.')
plt.loglog(np.arange(low1, high1+1), np.arange(low1, high1+1), '--', color='k')
plt.xlabel('mean deviant distance')
plt.ylabel('mean redundant distance')
plt.title('%s; Shape of contex dependent spaces' % rnn_leg[n_gr])


# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')


#%% distances with trials from qiet period
# plt.close('all')

zero_trials = 10
ymax = 100

n_gr = 0

for n_rnn in range(len(test_ob_all[n_gr][n_rnn])):

    f_plot_run_dist(test_ob_all[n_gr][n_rnn]['rates4d_cut'], plot_runs=50, plot_trials=100, zero_trials=zero_trials, stim_ave_win=[], run_labels = red_dd_seq[0,:], ymax = ymax, title_tag='ob inputs')

    f_plot_run_dist(test_cont_all[n_gr][n_rnn]['rates4d_cut'], plot_runs=20, plot_trials=100, zero_trials=zero_trials, stim_ave_win=[], run_labels = np.zeros((100)), ymax = ymax, title_tag='control inputs')


num_rnns = len(test_ob_all[n_gr])

run_labels = red_dd_seq[0,:]
run_labels_cont = np.zeros((100))
run_colors = cm.jet(np.linspace(0,1,np.max(run_labels)+1))
plot_runs=20
plot_trials=100
zero_trials=10

trial_len, num_trials, num_runs, num_cells = test_ob_all[0][0]['rates4d_cut'].shape

zero_trials2 = np.min((num_trials, zero_trials)).astype(int)
plot_trials2 = np.min((num_trials, plot_trials)).astype(int)
plot_runs2 = np.min((num_runs, plot_runs)).astype(int)

dist_all_ob = np.zeros((plot_trials2, plot_runs2, num_rnns))
for n_rnn in range(num_rnns):
    rates_in = test_ob_all[n_gr][n_rnn]['rates4d_cut']
    
    trial_len, num_trials, num_runs, num_cells = rates_in.shape
    
    rates_ave1 = np.mean(rates_in, axis=0)
    
    start_loc = np.mean(np.mean(rates_ave1[:zero_trials2,:,:], axis=0), axis=0)
    
    for n_run in range(plot_runs2):
        dist_all_ob[:,n_run, n_rnn] = np.sqrt(np.sum((rates_ave1[:plot_trials2,n_run,:] - start_loc)**2, axis=1))
        #dist1 = pdist(np.vstack((start_loc, rates3d_cut[:,0,:])))


dist_all_cont = np.zeros((plot_trials2, plot_runs2, num_rnns))
for n_rnn in range(num_rnns):
    rates_in = test_cont_all[n_gr][n_rnn]['rates4d_cut']
    
    trial_len, num_trials, num_runs, num_cells = rates_in.shape
    
    rates_ave1 = np.mean(rates_in, axis=0)
    
    start_loc = np.mean(np.mean(rates_ave1[:zero_trials2,:,:], axis=0), axis=0)
    
    for n_run in range(plot_runs2):
        dist_all_cont[:,n_run, n_rnn] = np.sqrt(np.sum((rates_ave1[:plot_trials2,n_run,:] - start_loc)**2, axis=1))
        #dist1 = pdist(np.vstack((start_loc, rates3d_cut[:,0,:])))


dist_all_ob2 = np.reshape(dist_all_ob, (plot_trials2, plot_runs2*num_rnns), order='F')
dist_all_cont2 = np.reshape(dist_all_cont, (plot_trials2, plot_runs2*num_rnns), order='F')

x_vals = np.arange(100)+1

dist_all_ob2_mean = np.mean(dist_all_ob2, axis=1)
dist_all_cont2_mean = np.mean(dist_all_cont2, axis=1)

dist_all_ob2_sem = np.std(dist_all_ob2, axis=1)/np.sqrt(plot_runs2*num_rnns-1)
dist_all_cont2_sem = np.std(dist_all_cont2, axis=1)/np.sqrt(plot_runs2*num_rnns-1)


plt.figure()
ax1 = f_plot_shadederrorbar2(x_vals, dist_all_ob2_mean, dist_all_ob2_sem, color='blue')
ax2 = f_plot_shadederrorbar2(x_vals, dist_all_cont2_mean, dist_all_cont2_sem, color='gray')
plt.title('network distance from quiet')
plt.ylabel('euclidean distance')
plt.xlabel('trials')
plt.legend(ax1+ax2, ['redundant', 'control'])

# f_save_fig(plt.figure(1), path=fig_path, name_tag='')

#%%
n_gr = 0

mean_mag_comb_dev = []
mean_mag_dev_dev = []
mean_mag_rd_dev = []
mean_mag_indiv_dev = []

mean_mag_comb_red = []
mean_mag_dev_red = []
mean_mag_rd_red = []
mean_mag_indiv_red = []

mean_cs_comb_dev = []
mean_cs_dev_dev = []
mean_cs_rd_dev = []

mean_cs_comb_red = []
mean_cs_dev_red = []
mean_cs_rd_red = []


mean_mag_comb_cont = []
mean_mag_freq_c_cont = []
mean_mag_freq_lc_cont = []
mean_mag_indiv_cont = []

mean_cs_comb_cont = []
mean_cs_freq_c_cont = []
mean_cs_freq_lc_cont = []

for n_rnn in range(len(trials_dev_all[n_gr])):
    
    #temp_indiv_mags, temp_mean_mag_rd, temp_mean_mag_d, temp_mean_mag_all = f_get_rd_vec_mags(trials_dev_all[n_gr][n_rnn]['indiv_trials'])
    
    mean_indiv_mags, mean_rd_mags, mean_dev_mags, mean_total_vec_mag, mean_rd_cossim, mean_dev_cossim, mean_total_cossim = f_get_rd_vec_mags2(trials_dev_all[n_gr][n_rnn]['indiv_trials'])
    
    # indiv_trials_rd = trials_dev_all[n_gr][n_rnn]['indiv_trials']
    
    mean_mag_comb_dev.append(mean_total_vec_mag)
    mean_mag_dev_dev.append(mean_dev_mags)
    mean_mag_rd_dev.append(mean_rd_mags)
    mean_mag_indiv_dev.append(mean_indiv_mags)
    
    mean_cs_comb_dev.append(mean_total_cossim)
    mean_cs_dev_dev.append(mean_dev_cossim)
    mean_cs_rd_dev.append(mean_rd_cossim)
    
    #temp_indiv_mags, temp_mean_mag_rd, temp_mean_mag_d, temp_mean_mag_all = f_get_rd_vec_mags(trials_red_all[n_gr][n_rnn]['indiv_trials'])
    mean_indiv_mags, mean_rd_mags, mean_dev_mags, mean_total_vec_mag, mean_rd_cossim, mean_dev_cossim, mean_total_cossim = f_get_rd_vec_mags2(trials_red_all[n_gr][n_rnn]['indiv_trials'])
    
    mean_mag_comb_red.append(mean_total_vec_mag)
    mean_mag_dev_red.append(mean_dev_mags)
    mean_mag_rd_red.append(mean_rd_mags)
    mean_mag_indiv_red.append(mean_indiv_mags)
    
    mean_cs_comb_red.append(mean_total_cossim)
    mean_cs_dev_red.append(mean_dev_cossim)
    mean_cs_rd_red.append(mean_rd_cossim)

    # indiv_trials_c = trials_cont_all[n_gr][n_rnn]['indiv_trials']
    #mean_indiv_mags, mean_c_mags, total_vec_mag, mean_c_cossim, mean_total_cossim = f_get_cont_vec_mags2(trials_cont_all[n_gr][n_rnn]['indiv_trials'])
    mean_indiv_mags, mean_lc_mags, mean_c_mags, mean_total_vec_mag, mean_lc_cossim, mean_c_cossim, mean_total_cossim = f_get_rd_vec_mags2(trials_cont2d_all[n_gr][n_rnn]['indiv_trials'])
    
    mean_mag_comb_cont.append(mean_total_vec_mag)
    mean_mag_freq_c_cont.append(mean_c_mags)
    mean_mag_freq_lc_cont.append(mean_lc_mags)
    mean_mag_indiv_cont.append(mean_indiv_mags)
    
    mean_cs_comb_cont.append(mean_total_cossim)
    mean_cs_freq_c_cont.append(mean_c_cossim)
    mean_cs_freq_lc_cont.append(mean_lc_cossim)


data_comb_d = np.array([mean_mag_comb_dev, mean_mag_dev_dev, mean_mag_rd_dev, mean_mag_indiv_dev])
data_comb_r = np.array([mean_mag_comb_red, mean_mag_dev_red, mean_mag_rd_red, mean_mag_indiv_red])
data_comb_cont = np.array([mean_mag_comb_cont, mean_mag_freq_c_cont, mean_mag_freq_lc_cont, mean_mag_indiv_cont])

dcd_mean = np.mean(data_comb_d, axis=1)
dcr_mean = np.mean(data_comb_r, axis=1)
dcc_mean = np.mean(data_comb_cont, axis=1)

max_val = np.max(np.hstack((dcd_mean, dcr_mean, dcc_mean))) # 

dcd_meann = dcd_mean/max_val
dcr_meann = dcr_mean/max_val
dcc_meann = dcc_mean/max_val


data_comb_cs_d = np.array([mean_cs_comb_dev, mean_cs_dev_dev, mean_cs_rd_dev])
data_comb_cs_r = np.array([mean_cs_comb_red, mean_cs_dev_red, mean_cs_rd_red])
data_comb_cs_c = np.array([mean_cs_comb_cont, mean_cs_freq_c_cont, mean_cs_freq_lc_cont])

dcd_cs_mean = np.mean(data_comb_cs_d, axis=1)
dcr_cs_mean = np.mean(data_comb_cs_r, axis=1)
dcc_cs_mean = np.mean(data_comb_cs_c, axis=1)


# plt.figure()
# plt.bar(['all' , 'red', 'red-dev', 'indiv'], np.mean(data_comb_d, axis=1))
# plt.ylabel('response mag') 
# plt.title('deviant')

# plt.figure()
# plt.bar(['all' , 'red', 'red-dev', 'indiv'], np.mean(data_comb_r, axis=1))
# plt.ylabel('response mag') 
# plt.title('redundant')

# plt.figure()
# plt.bar(['all', 'freq', 'indiv'], np.mean(data_comb_cont, axis=1))
# plt.ylabel('response mag') 
# plt.title('control')


shift1 = 0.3
width1 = 0.25
plt.figure()
plt.bar(['comb all' , 'split dev\ncomb red', 'split dev\nsplit red', 'indiv trials'], np.zeros(4))
plt.bar(np.arange(4)-shift1, dcr_meann, width=width1, color='blue')
plt.bar(np.arange(4), dcc_meann, width=width1, color='gray')
plt.bar(np.arange(4)+shift1, dcd_meann, width=width1, color='red')
plt.plot(np.arange(5)-0.5, np.ones(5), '--', color='k')
plt.ylabel('response magnitude') 
plt.title('%s; Vector magnitudes vs trial averaging' % rnn_leg[n_gr])



shift1 = 0.3
width1 = 0.25
plt.figure()
plt.bar(['comb all' , 'split dev\ncomb red', 'split dev\nsplit red'], np.zeros(3))
plt.bar(np.arange(3)-shift1,dcr_cs_mean, width=width1, color='blue')
plt.bar(np.arange(3), dcc_cs_mean, width=width1, color='gray')
plt.bar(np.arange(3)+shift1, dcd_cs_mean, width=width1, color='red')
plt.plot(np.arange(4)-0.5, np.ones(4), '--', color='k')
plt.ylabel('cosine similarity') 
plt.title('%s; Vector cosine similarity within trial' % rnn_leg[n_gr])


# f_save_fig(plt.figure(1), path=fig_path, name_tag='')
# f_save_fig(plt.figure(2), path=fig_path, name_tag='')


#%%
# plt.close('all')
mean_indiv_mag_dev_all = []
mean_indiv_mag_red_all = []

for n_rnn in range(len(trials_dev_all)):
    mean_indiv_mag_dev_all.append(trials_dev_all[n_rnn]['mean_indiv_mag'])
    mean_indiv_mag_red_all.append(trials_red_all[n_rnn]['mean_indiv_mag'])


mean_indiv_mag_dev_all = np.array(mean_indiv_mag_dev_all)
mean_indiv_mag_red_all = np.array(mean_indiv_mag_red_all)
    
for n_net in range(3):
    
    plt.figure()
    plt.imshow(np.mean(mean_indiv_mag_dev_all[net_idx==n_net], axis=0))
    plt.colorbar()
    plt.title('%s RNN dev trials' % rnn_leg[n_net])
    plt.ylabel('deviant freq')
    plt.xlabel('redundant freq')
    
    plt.figure()
    plt.imshow(np.mean(mean_indiv_mag_red_all[net_idx==n_net], axis=0))
    plt.colorbar()
    plt.title('%s RNN red trials' % rnn_leg[n_net])
    plt.ylabel('deviant freq')
    plt.xlabel('redundant freq')
    
    
    plt.figure()
    plt.imshow(np.mean(mean_indiv_mag_dev_all[net_idx==n_net], axis=0)/np.mean(mean_indiv_mag_red_all[net_idx==n_net], axis=0))
    plt.colorbar()
    plt.title('%s RNN dev-red ratio' % rnn_leg[n_net])
    plt.ylabel('deviant freq')
    plt.xlabel('redundant freq')
    

#%%#%% speed comparisons

speeds_dr = np.zeros((num_rnn, num_run, 2))

for n_rnn in range(num_rnn):
    test_data = test_ob_all[n_rnn]
    
    rates = test_data['rates_cut']
    
    
    for n_run in range(num_run):
        print('rnn %d; run %d' % (n_rnn, n_run))
        
        dist1 = squareform(pdist(rates[:,n_run,:], metric='euclidean'))
        dist2 = np.hstack((0,np.diag(dist1, 1)))
        
        dist23d = np.reshape(dist2, (trial_len, num_trials2), order = 'F')
        
        
        red_count_cut = ob_data1['red_count'][num_skip_trials:,:]
        dd_trial = np.where(red_count_cut[:,n_run] == 0)[0]
        last_red = dd_trial-1
        
        mean_dd_speed = np.mean(dist23d[5:15,dd_trial])
        mean_red_speed = np.mean(dist23d[5:15,last_red])
        
        speeds_dr[n_rnn, n_run, 0] = mean_dd_speed
        speeds_dr[n_rnn, n_run, 1] = mean_red_speed
 

speeds0_dr = np.zeros((num_rnn0, num_run, 2))

for n_rnn in range(num_rnn0):
    test_data = test0_ob_all[n_rnn]
    
    rates = test_data['rates_cut']

    for n_run in range(num_run):
        
        print('rnn %d; run %d' % (n_rnn, n_run))
        
        dist1 = squareform(pdist(rates[:,n_run,:], metric='euclidean'))
        dist2 = np.hstack((0,np.diag(dist1, 1)))
        
        dist23d = np.reshape(dist2, (trial_len, num_trials2), order = 'F')
        
        
        red_count_cut = ob_data1['red_count'][num_skip_trials:,:]
        dd_trial = np.where(red_count_cut[:,n_run] == 0)[0]
        last_red = dd_trial-1
        
        mean_dd_speed = np.mean(dist23d[5:15,dd_trial])
        mean_red_speed = np.mean(dist23d[5:15,last_red])
        
        speeds0_dr[n_rnn, n_run, 0] = mean_dd_speed
        speeds0_dr[n_rnn, n_run, 1] = mean_red_speed


#%%


plt.figure()
for n_rnn in range(num_rnn):
    norm_f = speeds_dr[n_rnn,:,0] + speeds_dr[n_rnn,:,1]
    plt.plot(speeds_dr[n_rnn,:,0], speeds_dr[n_rnn,:,1], '.')
plt.xlabel('dd mean speed')
plt.ylabel('red mean speed')
plt.xlim((np.min(speeds_dr[n_rnn,:,:])*0.95, np.max(speeds_dr[n_rnn,:,:])*1.05))
plt.ylim((np.min(speeds_dr[n_rnn,:,:])*0.95, np.max(speeds_dr[n_rnn,:,:])*1.05))
plt.title('trained')

plt.figure()
for n_rnn in range(num_rnn0):
    norm_f = speeds0_dr[n_rnn,:,0] + speeds0_dr[n_rnn,:,1]
    plt.plot(speeds0_dr[n_rnn,:,0], speeds0_dr[n_rnn,:,1], '.')
plt.xlabel('dd mean speed')
plt.ylabel('red mean speed')
plt.xlim((np.min(speeds0_dr[n_rnn,:,:])*0.95, np.max(speeds0_dr[n_rnn,:,:])*1.05))
plt.ylim((np.min(speeds0_dr[n_rnn,:,:])*0.95, np.max(speeds0_dr[n_rnn,:,:])*1.05))
plt.title('untrained')




