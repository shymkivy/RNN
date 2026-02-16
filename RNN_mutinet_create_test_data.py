# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:14:30 2024

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
from f_RNN import f_RNN_test, f_RNN_test_spont, f_gen_ob_dset, f_gen_cont_dset, f_RNN_load_multinet #, f_trial_ave_pad, f_gen_equal_freq_space
from f_RNN_process import f_trial_ave_ctx_pad, f_trial_ave_ctx_pad2, f_trial_sort_data_pad, f_trial_sort_data_ctx_pad, f_label_redundants, f_get_rdc_trav, f_gather_dev_trials, f_gather_red_trials, f_analyze_cont_trial_vectors, f_get_trace_tau, f_get_diags_data # , f_euc_dist, f_cos_sim , f_analyze_trial_vectors
from f_RNN_dred import f_run_dred, f_run_dred_wrap, f_proj_onto_dred
from f_RNN_plots import f_plot_dred_rates, f_plot_dred_rates2, f_plot_dred_rates3, f_plot_dred_rates3d, f_plot_traj_speed, f_plot_resp_distances, f_plot_mmn, f_plot_mmn2, f_plot_mmn_dist, f_plot_mmn_freq, f_plot_dred_pcs, f_plot_rnn_weights2, f_plot_run_dist, f_plot_cont_vec_data, f_plot_rd_vec_data, f_plot_ctx_vec_data, f_plot_ctx_vec_dir, f_plot_cat_data # , f_plot_shadederrorbar
from f_RNN_chaotic import RNN_chaotic
from f_RNN_utils import f_gen_stim_output_templates, f_gen_cont_seq, f_gen_oddball_seq, f_gen_input_output_from_seq, f_plot_examle_inputs, f_plot_train_loss, f_plot_train_test_loss, f_gen_name_tag, f_cut_reshape_rates_wrap, f_plot_exp_var, f_plot_freq_space_distances_control, f_plot_freq_space_distances_oddball # , f_reshape_rates
from f_RNN_decoder import f_make_cv_groups, f_sample_trial_data_dec, f_run_binwise_dec, f_plot_binwise_dec, f_run_one_shot_dec, f_plot_one_shot_dec_bycat, f_plot_one_shot_dec_bycat2, f_plot_one_shot_dec_iscat

import time
import numpy as np
import matplotlib.pyplot as plt
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
from scipy import linalg
#from scipy.io import loadmat, savemat
#import skimage.io

from datetime import datetime



#%%

# rnn_flist = ['oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_20trials_50stim_100batch_1e-03lr_2023_9_11_14h_19m_RNN',
#              'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_4_17h_16m_RNN',
#              'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_5_10h_54m_RNN',
#              'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_6_17h_3m_RNN']

rnn_flist = [
             'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_9_11_14h_19m_ext_2024_3_6_16h_56m_RNN',
               #'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_9_11_14h_19m_RNN',
               'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_10_4_17h_16m_ext_2024_3_7_12h_9m_RNN',
               #'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_4_17h_16m_RNN',
               #'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_10_5_10h_54m_ext_2024_3_8_11h_57m_RNN',   # explodes
               #'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_5_10h_54m_RNN',
               'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_2023_10_6_17h_3m_RNN',
               'oddball2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_19_16h_20m_RNN',
               #'oddball2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_5_13h_15m_RNN', # 250 neurons
               #'oddball2_1ctx_120000trainsamp_250neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_22_12h_49m_RNN', # not long enough? 250 neurons dt 10 cant use different dt
               #'oddball2_1ctx_120000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_1_11h_45m_RNN', # not long enough? 100 neurons # dt 10
               #'oddball2_1ctx_180000trainsamp_100neurons_ReLU_100tau_10dt_20trials_50stim_100batch_4e-05lr_noise1_2024_2_4_20h_33m_RNN', # not long enough? 100 neurons # dt 10
               'oddball2_1ctx_140000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_19_10h_15m_RNN',
               'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_24_13h_23m_RNN', #didn't reach low enough loss
               'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_28_11h_32m_RNN', #didn't reach low enough loss
               #'oddball2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_3_13h_54m_RNN', #didn't reach low enough loss   # explodes
               #'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-03lr_noise1_2024_3_6_16h_3m_RNN',    # explodes
               'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_7_12h_8m_RNN',
               'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_8_11h_57m_RNN',
               'oddball2_1ctx_200000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_4h_38m_RNN',
               'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_10_21h_27m_RNN',
               'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_11_14h_58m_RNN',
               'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_2e-03lr_noise1_2024_3_11_16h_47m_RNN',
               'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_13_11h_26m_RNN',
              #'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_14_10h_45m_RNN',    # explodes with preappended zeros
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_15_10h_58m_RNN',
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_50batch_1e-03lr_noise1_2024_3_17_21h_14m_RNN',
              'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_50batch_1e-03lr_noise1_2024_3_18_10h_24m_RNN',
              'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_50batch_1e-03lr_noise1_2024_3_19_18h_29m_RNN',    # start of new things
              'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_50batch_1e-03lr_noise1_2024_3_20_17h_11m_RNN',
              'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_50batch_1e-03lr_noise1_2024_3_21_19h_52m_RNN',
              'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_50batch_1e-03lr_noise1_2024_3_22_11h_44m_RNN',
              'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_24_13h_59m_RNN',
              'oddball2_1ctx_200000trainsamp_150neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_25_17h_41m_RNN',
              'oddball2_1ctx_200000trainsamp_200neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_26_18h_34m_RNN',
              'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_75batch_1e-03lr_noise1_2024_3_27_22h_55m_RNN',
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_28_18h_28m_RNN',
              'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_29_19h_1m_RNN',
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_3_30_15h_56m_RNN',
              'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_1_22h_53m_RNN',
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_3_17h_52m_RNN',
              'oddball2_1ctx_200000trainsamp_50neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_4_18h_56m_RNN',
              'oddball2_1ctx_200000trainsamp_50neurons_tanh_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_5_13h_12m_RNN',
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_8_17h_18m_RNN',
              'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_9_19h_42m_RNN',
              'oddball2_1ctx_200000trainsamp_100neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_11_17h_9m_RNN',
              'oddball2_1ctx_200000trainsamp_75neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_4_12_17h_33m_RNN',
             ]

rnnf_flist = ['freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise0_2023_12_20_0h_34m_RNN',
               'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2023_12_20_0h_34m_RNN',
               'freq2_1ctx_80000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_1_4_13h_14m_RNN', # not long enough?
               #'freq2_1ctx_80000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-04lr_noise1_2024_1_10_11h_28m_RNN', # bit spiky
               #'freq2_1ctx_120000trainsamp_250neurons_ReLU_500tau_50dt_20trials_50stim_100batch_4e-05lr_noise1_2024_1_11_11h_33m_RNN', # bit spiky
               'freq2_1ctx_160000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_1e-03lr_noise1_2024_2_20_19h_9m_RNN', # bit spiky
               'freq2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_22_16h_20m_RNN',
               'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_2_27_13h_17m_RNN',
               'freq2_1ctx_250000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_1_10h_1m_RNN',
               'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_4_19h_25m_RNN',
               'freq2_1ctx_300000trainsamp_25neurons_ReLU_500tau_50dt_20trials_50stim_100batch_5e-04lr_noise1_2024_3_5_0h_28m_RNN',
              ]

#%%
params_test = {'num_prepend_zeros':         100,
               'num_dev_stim':              20,
               'num_red_stim':              20,
               'num_cont_stim':             20,
               'num_ob_runs':               400,
               'num_ob_trials':             200,
               'num_cont_runs':             50,
               'num_cont_trials':           200,
               
               'dd_frac':                   0.1,
               
               'input_size':                50,
               'num_freq_stim':             50,
               'stim_duration':             0.5,
               'isi_duration':              0.5,
               'dt':                        0.05,
               
               'input_noise_std':           1/100,
               'normalize_input':           False,
               'stim_t_std':                3,
               
               'plot_deets':                False,
               }


#%% load all params and rnns

flist_all = [rnn_flist, rnnf_flist, []]
untrain_param_source = [0, 0, 1]
data_path = 'F:/RNN_stuff/RNN_data/'

rnn_leg = ['ob trained', 'freq trained', 'untrained']

rnn_all, params_all, net_idx = f_RNN_load_multinet(flist_all, data_path, untrain_param_source, max_untrained = 20)     

#%%
stim_templates = f_gen_stim_output_templates(params_test)
trial_len = round((params_test['stim_duration'] + params_test['isi_duration'])/params_test['dt'])

#%% create test inputs

ob_data1 = f_gen_ob_dset(params_test, stim_templates, num_trials=params_test['num_ob_trials'], num_runs=params_test['num_ob_runs'], num_dev_stim=params_test['num_dev_stim'], num_red_stim=params_test['num_red_stim'], num_freqs=params_test['num_freq_stim'], stim_sample='equal', ob_type='one_deviant', freq_selection='sequential', can_be_same = False, can_have_no_dd = True, prepend_zeros=params_test['num_prepend_zeros'])       # stim_sample= 'random' or 'equal'; ob_type='one_deviant' or 'many_deviant', '100plus1'
cont_data = f_gen_cont_dset(params_test, stim_templates, num_trials=params_test['num_cont_trials'], num_runs=params_test['num_cont_runs'], num_cont_stim=params_test['num_cont_stim'], num_freqs=params_test['num_freq_stim'], prepend_zeros=params_test['num_prepend_zeros'])

#%% 
test_ob_all = []
test_cont_all = []
for n_tr in range(len(rnn_all)):
    test_ob2 = []
    test_cont2 = []
    for n_rnn in range(len(rnn_all[n_tr])):
        print('%s, rnn %d of %d' % (rnn_leg[n_tr], n_rnn+1, len(rnn_all[n_tr])))
        
        loss_freq = nn.CrossEntropyLoss().cpu()
        loss_ctx = nn.CrossEntropyLoss(weight = torch.tensor(params_all[n_tr][n_rnn]['train_loss_weights']).to('cpu'))

        test_oddball_ctx = f_RNN_test(rnn_all[n_tr][n_rnn], loss_ctx, ob_data1['input_oddball'], ob_data1['target_oddball_ctx'], paradigm='ctx')
        test_ob2.append(test_oddball_ctx)
        test_cont_freq = f_RNN_test(rnn_all[n_tr][n_rnn], loss_freq, cont_data['input_control'], cont_data['target_control'], paradigm='freq')
        test_cont2.append(test_cont_freq)
    test_ob_all.append(test_ob2)
    test_cont_all.append(test_cont2)

#%% and save

now1 = datetime.now()
date_tag_ext = '_%d_%d_%d_%dh_%dm' % (now1.year, now1.month, now1.day, now1.hour, now1.minute)
fname_RNN_save = 'test_data%s' % (date_tag_ext)

data_all = {
            'params_all':               params_all,
            'rnn_leg':                  rnn_leg,
            'params_test':              params_test,
            'net_idx':                  net_idx,
            'flist_all':                flist_all,
            'untrain_param_source':     untrain_param_source,
            'ob_data':                  ob_data1,
            'cont_data':                cont_data,
            }

data1 = {'test_ob_all':     test_ob_all}
data2 = {'test_cont_all':     test_cont_all}

np.save(data_path + '/test_data/' + fname_RNN_save + '_params.npy', data_all) 
np.save(data_path + '/test_data/' + fname_RNN_save + '_cont_data.npy', data2) 
#del test_cont_all
#del data2
#del cont_data
#del test_cont_freq

np.save(data_path + '/test_data/' + fname_RNN_save + '_ob_data.npy', data1) 


#np.savez(data_path + '/test_data/' + fname_RNN_save, data1=data1, data2=data2, data_all=data_all) 





