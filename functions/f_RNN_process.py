# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:09:22 2024

@author: ys2605
"""

import numpy as np

from scipy.spatial.distance import pdist, squareform, cdist #
from scipy.signal import correlate #, correlation_lags

#import matplotlib.pyplot as plt

#%%

def f_plot_t(trial_len, dt):
    plot_t = (np.arange(trial_len)+1-trial_len/4)*dt
    return plot_t

def f_get_stim_on_bins(trial_len, params):
    trial_stim_on = np.zeros(trial_len, dtype=bool)
    trial_stim_on[round(np.floor(params['isi_duration']/params['dt']/2)):(round(np.floor(params['isi_duration']/params['dt']/2))+round(params['stim_duration']/params['dt']))] = 1
    return trial_stim_on

def f_get_diags_data(data_in):

    num_freq = data_in.shape[0]
    x_range = np.arange(-num_freq+1, num_freq)
    y_means = np.zeros(num_freq*2-1)
    y_stds = np.zeros(num_freq*2-1)
    
    for n_freq in range(num_freq*2-1):
        dist_vals = np.diagonal(data_in, offset=x_range[n_freq])
        y_means[n_freq] = np.mean(dist_vals)
        y_stds[n_freq] = np.std(dist_vals)
    
    return x_range, y_means, y_stds


def f_get_cs_same_dev(data_rd):
    
    num_red, num_dev, num_cells = data_rd.shape
    cos_sim_ref_dev = np.zeros((num_red, num_red, num_dev))
    
    for n_freq in range(num_dev): # for each deviant freq, compare responses in red
        dist1 = squareform(pdist(data_rd[:,n_freq,:], 'cosine'))
        cos_sim_ref_dev[:,:,n_freq] = 1 - dist1
    
    cos_sim_ref_dev2 = np.mean(cos_sim_ref_dev, axis=2)
    
    return cos_sim_ref_dev2

def f_get_cs_same_red(data_rd):
    
    num_red, num_dev, num_cells = data_rd.shape
    cos_sim_ref_red = np.zeros((num_dev, num_dev, num_red))
    
    for n_freq in range(num_red): # for each deviant freq, compare responses in red
        dist1 = squareform(pdist(data_rd[n_freq,:,:], 'cosine'))
        cos_sim_ref_red[:,:,n_freq] = 1 - dist1
    
    cos_sim_ref_red2 = np.mean(cos_sim_ref_red, axis=2)
    
    return cos_sim_ref_red2


def f_get_cs_cont_to_ref_same_dev(data_rd, data_cont):
    
    num_red, num_dev, num_cells = data_rd.shape
    cos_sim_cont_dev = np.zeros((num_red, num_red, num_dev))
    
    for n_freq in range(num_dev): # for each deviant freq, compare responses in red
        dist1 = cdist(data_rd[:,n_freq,:], data_cont, 'cosine')
        cos_sim_cont_dev[:,:,n_freq] = 1 - dist1
    
    cos_sim_cont_dev2 = np.mean(cos_sim_cont_dev, axis=2)
    
    return cos_sim_cont_dev2 #, cos_sim_cont_dev

def f_get_cs_cont_to_ref_same_red(data_rd, data_cont):
    
    num_red, num_dev, num_cells = data_rd.shape
    cos_sim_cont_red = np.zeros((num_dev, num_dev, num_red))
    
    for n_freq in range(num_red): # for each deviant freq, compare responses in red
        dist1 = cdist(data_rd[n_freq,:,:], data_cont, 'cosine')
        cos_sim_cont_red[:,:,n_freq] = 1 - dist1
    
    cos_sim_cont_red2 = np.mean(cos_sim_cont_red, axis=2)
    
    return cos_sim_cont_red2

def f_get_cs_cont_to_dev_vs_red(data_rd, data_cont):
    
    num_red, num_dev, num_cells = data_rd.shape
    cos_sim_cont_dev = np.zeros((num_red, num_red))
    
    for n_freq in range(num_dev): # for each deviant freq, compare responses in red
        dist1 = cdist(data_rd[:,n_freq,:], data_cont, 'cosine')
        cos_sim_cont_dev[:,n_freq] = 1 - dist1[:,n_freq]
    
    #cos_sim_cont_dev2 = np.mean(cos_sim_cont_dev, axis=2)
    
    return cos_sim_cont_dev #, cos_sim_cont_dev


#%%

def f_label_redundants(trials_ctx):
    
    num_trials, num_runs = trials_ctx.shape
    
    num_ctx = np.unique(trials_ctx).shape[0]
    
    if num_ctx == 2:
        red_idx = 0
    elif num_ctx == 3:
        red_idx = 1
    
    forward_label = np.zeros((num_trials, num_runs), dtype=int)
    reverse_label = np.zeros((num_trials, num_runs), dtype=int)
    
    for n_run in range(num_runs):
        
        n_red = 0
        for n_tr in range(num_trials):
            if trials_ctx[n_tr, n_run] == red_idx: # if red
                n_red += 1
            else:
                n_red = 0
            forward_label[n_tr, n_run] = n_red
            
        n_red = -99
        for n_tr in range(num_trials):
            n_tr_rev = num_trials-n_tr-1
            
            if trials_ctx[n_tr_rev, n_run] == red_idx: # if red
                n_red -= 1
            else:
                n_red=0
            reverse_label[n_tr_rev, n_run] = n_red
                  
    return forward_label, reverse_label


#%%

def f_get_rdc_trav(trials_oddball_ctx_cut, rates4d_cut, trials_cont_cut, rates_cont_freq4d_cut, params, red_dd_seq, baseline_subtract=True, base_time = [-0.5, 0]):
    trial_len, num_trials, num_runs, num_cells = rates4d_cut.shape
    _, _, num_runs_cont, _ = rates_cont_freq4d_cut.shape
    
    
    
    plot_t1 = f_plot_t(trial_len, params['dt'])
    base_idx = np.logical_and(plot_t1>base_time[0], plot_t1<base_time[1])

    num_cont_trials, num_cont_runs = trials_cont_cut.shape

    trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut)


    freqs_all = np.unique(red_dd_seq)
    num_freqs = freqs_all.shape[0]

    trial_ave_rdc = np.zeros((trial_len, 3, num_freqs, num_cells))

    for n_freq in range(num_freqs):
        freq1 = freqs_all[n_freq]
        
        red_run_idx = red_dd_seq[0,:] == freq1
        dev_run_idx = red_dd_seq[1,:] == freq1
        
        red_resp_all = []
        dev_resp_all = []
        for n_run in range(num_runs):
            if red_run_idx[n_run]:
                #red_tr_idx = trials_oddball_red_fwr[:,n_run] == 3
                red_tr_idx = trials_oddball_red_rev[:,n_run] == -3
                red_resp_all.append(rates4d_cut[:,red_tr_idx, n_run,:])
            
            if dev_run_idx[n_run]:
                dev_tr_idx = trials_oddball_ctx_cut[:,n_run] == 1
                dev_resp_all.append(rates4d_cut[:,dev_tr_idx, n_run,:])
        
        red_resp_all2 = np.concatenate(red_resp_all, axis=1)
        dev_resp_all2 = np.concatenate(dev_resp_all, axis=1)
        
        cont_resp_all = []
        for n_run in range(num_runs_cont):
            cont_tr_idx = trials_cont_cut[:,n_run] == freq1
            cont_resp_all.append(rates_cont_freq4d_cut[:,cont_tr_idx,n_run,:])
        
        cont_resp_all2 = np.concatenate(cont_resp_all, axis=1)
        
        red_tr_ave = np.mean(red_resp_all2, axis=1)
        dev_tr_ave = np.mean(dev_resp_all2, axis=1)
        cont_tr_ave = np.mean(cont_resp_all2, axis=1)
      
        if baseline_subtract:
            red_tr_ave3 = red_tr_ave - np.mean(red_tr_ave[base_idx,:], axis=0)
            dev_tr_ave3 = dev_tr_ave - np.mean(dev_tr_ave[base_idx,:], axis=0)
            cont_tr_ave3 = cont_tr_ave - np.mean(cont_tr_ave[base_idx,:], axis=0)
        else:
            red_tr_ave3 = red_tr_ave
            dev_tr_ave3 = dev_tr_ave
            cont_tr_ave3 = cont_tr_ave
        
        trial_ave_rdc[:,0,n_freq,:] = red_tr_ave3
        trial_ave_rdc[:,1,n_freq,:] = dev_tr_ave3
        trial_ave_rdc[:,2,n_freq,:] = cont_tr_ave3
        
    return trial_ave_rdc

#%%
def f_trial_ave_ctx_pad(rates4d_cut, trials_types_cut, pre_dd = 2, post_dd = 2):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    for n_run in range(num_run):
            
        temp_ob = trials_types_cut[:,n_run]
        
        temp_sum1 = np.zeros((num_t, num_tr_ave, num_cells))
        num_dd = 0
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]:
                
                if np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1:
                        
                    num_dd += 1
                    temp_sum1 += rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                
        temp_ave4d[:,:,n_run,:] = temp_sum1/num_dd
        
        trial_ave3d = np.reshape(temp_ave4d, (num_t*num_tr_ave, num_run, num_cells), order = 'F')
        
    return trial_ave3d

#%%
def f_trial_ave_ctx_pad2(rates4d_cut, trials_types_cut, pre_dd = 2, post_dd = 2, limit_1_dd=False, max_trials=999, shuffle_trials=False):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    max_dd_trials = np.max(np.sum(trials_types_cut, axis=0))
    
    num_dd_trials = np.zeros((num_run), dtype=int)
    trial_data_sort = []
    
    for n_run in range(num_run):
            
        temp_ob = trials_types_cut[:,n_run]
        
        num_dd=0
        
        trial_data_sort2 = []
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]: # if currently on dd trial
                
                if limit_1_dd:  # if only 1 dd in vicinity
                    get_trial = np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1
                else:
                    get_trial = True
            
                if get_trial:
                    
                    trial_data_sort2.append(rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :][:,:,None,:])
                    num_dd += 1

        
        trial_data_sort3 = np.concatenate(trial_data_sort2, axis=2)
        
        if shuffle_trials:
            shuff_idx = np.arange(num_dd)
            np.random.shuffle(shuff_idx)
            trial_data_sort3 = trial_data_sort3[:,:,shuff_idx,:]
        
        use_dd = np.min((num_dd,max_trials))
        
        trial_data_sort4 = trial_data_sort3[:,:,:use_dd,:]
        
        num_dd_trials[n_run] = use_dd
        
        temp_ave4d[:,:,n_run,:] = np.mean(trial_data_sort4,axis=2)
        
        trial_data_sort.append(trial_data_sort4)
    
    return temp_ave4d, trial_data_sort, num_dd_trials


#%%
def f_trial_sort_data_pad(rates4d_cut, trials_types_cut, pre_trials = 2, post_trials = 2):
    # get trial ave and sorted single trial data with more trials
    
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_trials + post_trials + 1
    
    trials_per_run = num_tr - pre_trials - post_trials
    
    trial_data_sort = np.zeros((num_t, num_tr_ave, trials_per_run, num_run, num_cells))
    trial_types_pad = np.zeros((num_tr_ave, trials_per_run, num_run))
    for n_run in range(num_run):
            
        for n_tr in range(pre_trials, num_tr-post_trials-1):
            trial_data_sort[:,:,n_tr-pre_trials,n_run,:] = rates4d_cut[:,(n_tr-pre_trials):(n_tr+post_trials+1),n_run,:]
            trial_types_pad[:,n_tr-pre_trials,n_run] = trials_types_cut[(n_tr-pre_trials):(n_tr+post_trials+1),n_run]
    
    if post_trials:
        trials_types_out = trials_types_cut[pre_trials:-post_trials,:]
    else:
        trials_types_out = trials_types_cut[pre_trials:,:]
    
    trial_data_sort4d = np.reshape(trial_data_sort, (num_t*num_tr_ave, trials_per_run, num_run, num_cells), order = 'F')
    trial_types_pad2d = np.reshape(trial_types_pad, (num_tr_ave, trials_per_run, num_run), order = 'F')
    
    return trial_data_sort4d, trials_types_out, trial_types_pad2d

#%%
def f_trial_sort_data_ctx_pad(rates4d_in, trials_types_ctx, trials_types_freq, pre_trials = 2, post_trials = 2, max_trials=999, shuffle_trials=False):
    # get trial ave and sorted single trial data with more trials
    # zero  trials get thrown away
    
    num_t, num_tr, num_run, num_cells = rates4d_in.shape
    num_tr_ave = pre_trials + post_trials + 1
    
    #trials_per_run = num_tr - pre_trials - post_trials
    
    trial_data_sort = []
    dd_freqs_out = []
    num_dd_trials_all = np.zeros((num_run), dtype=int)
  
    for n_run in range(num_run):
        
        num_dd_trials = np.sum(trials_types_ctx[pre_trials:-post_trials-1, n_run]).astype(int)
        
        trial_data_sort2 = np.zeros((num_t, num_tr_ave, num_dd_trials, num_cells))
        dd_type_freq = np.zeros(num_dd_trials, dtype=int)
        
        n_dd = 0
        for n_tr in range(pre_trials, num_tr-post_trials-1):
            if trials_types_ctx[n_tr, n_run]:
                trial_data_sort2[:,:,n_dd,:] = rates4d_in[:,(n_tr-pre_trials):(n_tr+post_trials+1),n_run,:]
                dd_type_freq[n_dd] = trials_types_freq[n_tr, n_run]
                n_dd += 1
                

        if shuffle_trials:
            shuff_idx = np.arange(num_dd_trials)
            np.random.shuffle(shuff_idx)
            trial_data_sort3 = trial_data_sort2[:, :, shuff_idx, :]
            dd_type_freq2 = dd_type_freq[shuff_idx]
        else:
            trial_data_sort3 = trial_data_sort2
            dd_type_freq2 = dd_type_freq
        
        num_dd_trials2 = np.min((num_dd_trials, max_trials))
        
        trial_data_sort4 = trial_data_sort3[:,:,:num_dd_trials2,:]
        dd_type_freq3 = dd_type_freq2[:num_dd_trials2]
        
        trial_data_sort4_3d = np.reshape(trial_data_sort4, (num_t*num_tr_ave, num_dd_trials2, num_cells), order = 'F')
        
        num_dd_trials_all[n_run] = num_dd_trials2
        
        trial_data_sort.append(trial_data_sort4_3d)
        dd_freqs_out.append(dd_type_freq3)
        
    return trial_data_sort, dd_freqs_out, num_dd_trials_all

#%%
def f_trial_ave_pad(rates4d_cut, trials_types_cut, pre_dd = 2, post_dd = 2):
    # get trial ave and sorted single trial data with more trials
    
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    num_tr_ave = pre_dd + post_dd + 1
    
    temp_ave4d = np.zeros((num_t, num_tr_ave, num_run, num_cells))
    
    max_dd_trials = np.max(np.sum(trials_types_cut, axis=0))
    
    num_dd_trials = np.zeros((num_run))
    trial_data_sort = np.zeros((num_t, num_tr_ave, max_dd_trials, num_run, num_cells))
    
    for n_run in range(num_run):
            
        temp_ob = trials_types_cut[:,n_run]
        
        temp_sum1 = np.zeros((num_t, num_tr_ave, num_cells))
        num_dd = 0
        
        for n_tr in range(pre_dd, num_tr-post_dd):
            
            if temp_ob[n_tr]: # if currently on dd trial
                
                if np.sum(temp_ob[n_tr-pre_dd:n_tr+post_dd+1]) == 1: # if only 1 dd in vicinity
                    
                    trial_data_sort[:,:,num_dd,n_run,:] = rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                    num_dd_trials[n_run] +=1
                    
                    num_dd += 1
                    temp_sum1 += rates4d_cut[:, n_tr-pre_dd:n_tr+post_dd+1, n_run, :]
                    
        temp_ave4d[:,:,n_run,:] = temp_sum1/num_dd
    
    return temp_ave4d, trial_data_sort, num_dd_trials

#%%
def f_trial_ave_ctx_rd(rates4d_cut, trials_types_cut, params):
    num_t, num_tr, num_run, num_cells = rates4d_cut.shape
    
    if params['num_ctx'] == 1:
        ctx_pad1 = 0
    elif params['num_ctx'] == 2:
        ctx_pad1 = 1
        
    trial_ave_rd = np.zeros((2, num_t, num_run, num_cells))
    
    for n_run in range(num_run):
        idx1 = trials_types_cut[:,n_run] == 0+ctx_pad1
        trial_ave_rd[0,:,n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
        
        idx1 = trials_types_cut[:,n_run] == 1+ctx_pad1
        trial_ave_rd[1,:,n_run,:] = np.mean(rates4d_cut[:,idx1,n_run,:], axis=1)
    
    return trial_ave_rd

#%%
def f_euc_dist(vec1, vec2):
    
    dist_sq = (vec1 - vec2)**2
    
    if len(dist_sq.shape) == 1:
        dist1 = np.sqrt(np.sum(dist_sq, axis=0))
    else:
        dist1 = np.sqrt(np.sum(dist_sq, axis=1))
    
    return dist1

def f_cos_sim(vec1, vec2):
    
    vec1_mag = np.sqrt(np.sum(np.abs(vec1.T)**2, axis=1))
    vec2_mag = np.sqrt(np.sum(np.abs(vec2)**2, axis=0))
    
    vec_angles = np.dot(vec1.T, vec2)/(vec1_mag*vec2_mag)
    
    return vec_angles


#%%
def f_gather_dev_trials(rates4d_cut, trials_oddball_ctx_cut, red_dd_seq):

    num_trials, num_runs = trials_oddball_ctx_cut.shape
    
    freq_red_all = np.unique(red_dd_seq[0,:])
    freqs_dev_all = np.unique(red_dd_seq[1,:])
    num_freq_r = len(freq_red_all)
    num_freq_d = len(freqs_dev_all)
    
    trials_rd = []
    for n_fr in range(num_freq_r):
        trials_d = []
        for n_fd in range(num_freq_d):
            trials_d.append([])
        trials_rd.append(trials_d)
    
    for n_run in range(num_runs):
        freq_red = red_dd_seq[0,n_run]
        freq_dev = red_dd_seq[1,n_run]
        
        red_loc = np.where(freq_red_all==freq_red)[0][0]
        dev_loc = np.where(freqs_dev_all==freq_dev)[0][0]
        
        #dd_idx = trials_oddball_freq_cut[:,n_run] == freq_dev
        dd_idx = trials_oddball_ctx_cut[:,n_run] == 1
        
        temp_rates = rates4d_cut[:,dd_idx,n_run,:]
        
        trials_rd[red_loc][dev_loc].append(temp_rates)
    
    return trials_rd

def f_gather_red_trials(rates4d_cut, trials_oddball_freq_cut, trials_oddball_ctx_cut, red_dd_seq, red_idx = 3):

    num_trials, num_runs = trials_oddball_freq_cut.shape
    
    freq_red_all = np.unique(red_dd_seq[0,:])
    freqs_dev_all = np.unique(red_dd_seq[1,:])
    num_freq_r = len(freq_red_all)
    num_freq_d = len(freqs_dev_all)
    
    trials_rd = []
    for n_fr in range(num_freq_r):
        trials_d = []
        for n_fd in range(num_freq_d):
            trials_d.append([])
        trials_rd.append(trials_d)
    
    for n_run in range(num_runs):
        freq_red = red_dd_seq[0,n_run]
        freq_dev = red_dd_seq[1,n_run]

        red_loc = np.where(freq_red_all==freq_red)[0][0]
        dev_loc = np.where(freqs_dev_all==freq_dev)[0][0]
        
        trials_oddball_freq_cut2 = trials_oddball_freq_cut[:,n_run]
        trials_oddball_ctx_cut2 = trials_oddball_ctx_cut[:,n_run]
        trials_oddball_ctx_cut3 = trials_oddball_ctx_cut2[trials_oddball_freq_cut2>0]
        
        temp_rates = rates4d_cut[:,:,n_run,:]
        temp_rates2 = temp_rates[:,trials_oddball_freq_cut2>0,:]
        
        trials_oddball_red_fwr, trials_oddball_red_rev = f_label_redundants(trials_oddball_ctx_cut3[:,None])
        
        if red_idx>0:
            red_idx2 = trials_oddball_red_fwr[:,0] == red_idx
        else:
            red_idx2 = trials_oddball_red_rev[:,0] == red_idx
                
        temp_rates3 = temp_rates2[:,red_idx2,:]
        
        trials_rd[red_loc][dev_loc].append(temp_rates3)
    
    return trials_rd


def f_gather_cont_trials(rates_cont4d, trials_cont_cut, freqs_list):

    trial_len, _, num_runs, num_cells = rates_cont4d.shape
    
    _, num_runs = trials_cont_cut.shape
    
    freqs_all = np.unique(freqs_list)
    num_freqs = freqs_all.shape[0]
    
    trials_all = []
    
    for n_freq in range(num_freqs):
        freq1 = freqs_all[n_freq]
        trials_all2 = []
        for n_run in range(num_runs):
            tr_idx = trials_cont_cut[:,n_run] == freq1
        
            temp_data = rates_cont4d[:,tr_idx,n_run,:]
            
            trials_all2.append(temp_data)
            
        trials_all.append(np.concatenate(trials_all2, axis=1))
    
    return trials_all

def f_gather_cont_trials2d(rates_cont4d, trials_cont_cut, freqs_list):
    # also sort by history (last tr id, cur tr id)
    trial_len, _, num_runs, num_cells = rates_cont4d.shape
    
    _, num_runs = trials_cont_cut.shape
    
    freqs_all = np.unique(freqs_list)
    num_freqs = freqs_all.shape[0]
    
    trials_all = []
    
    for n_freq1 in range(num_freqs):
        trials_all1 = []
        freq1 = freqs_all[n_freq1]
        for n_freq2 in range(num_freqs):
            freq2 = freqs_all[n_freq2]
            
            trials_all2 = []
            
            for n_run in range(num_runs):
                tr_idx1 = trials_cont_cut[:,n_run] == freq1
                tr_idx2 = trials_cont_cut[:,n_run] == freq2
                
                tr_idx = np.hstack((False, tr_idx1[:-1]*tr_idx2[1:]))
                
                if np.sum(tr_idx):
                    temp_data = rates_cont4d[:,tr_idx,n_run,:]
                    trials_all2.append(temp_data)
                
            trials_all1.append(trials_all2)
        trials_all.append(trials_all1)
    
    return trials_all


#%%

def f_get_vec_data(vec_in, base_idx, on_idx):

    # baselibes and on of each trial
    base_trials = np.mean(vec_in[base_idx,:,:], axis=0)
    on_trials = np.mean(vec_in[on_idx,:,:], axis=0)
    
    # base and on trial ave
    base_mean = np.mean(base_trials, axis=0)
    on_mean = np.mean(on_trials, axis=0)
    
    base_std = np.std(base_trials, axis=0)
    on_std = np.std(on_trials, axis=0)
    
    mean_vec = on_mean - base_mean
    mean_vec_mag = f_euc_dist(base_mean, on_mean)

    # distances from mean on each trial
    base_dist_trials = f_euc_dist(base_trials, base_mean)
    on_dist_trials = f_euc_dist(on_trials, on_mean)
    
    base_dist_mean = np.mean(base_dist_trials)
    on_dist_mean = np.mean(on_dist_trials)
    
    base_dist_std = np.std(base_dist_trials)
    on_dist_std = np.std(on_dist_trials)
    
    indiv_trials_mag =  f_euc_dist(base_trials, on_trials)
    indiv_trial_mag_mean = np.mean(indiv_trials_mag)
    indiv_trial_mag_std = np.std(indiv_trials_mag)
    
    indiv_trials_vec = on_trials - base_trials
    indiv_trials_angles = f_cos_sim(indiv_trials_vec.T, mean_vec)
    #vec_angles2 = 1 - pdist(np.vstack((mean_direction, trial_directions)), 'cosine')
    
    indiv_trials_angles_mean = np.mean(indiv_trials_angles)
    indiv_trials_angles_std = np.std(indiv_trials_angles)
    
    vec_data = {'base_mean':                base_mean,
                'on_mean':                  on_mean,
                'base_std':                 base_std,
                'on_std':                   on_std,
                'mean_vec':                 mean_vec,
                'mean_vec_mag':             mean_vec_mag,
                'base_dist_mean':           base_dist_mean,
                'on_dist_mean':             on_dist_mean,
                'base_dist_std':            base_dist_std,
                'on_dist_std':              on_dist_std,
                'indiv_trial_mag_mean':     indiv_trial_mag_mean,
                'indiv_trial_mag_std':      indiv_trial_mag_std,
                'indiv_trial_angles_mean':  indiv_trials_angles_mean,
                'indiv_trial_angles_std':   indiv_trials_angles_std,
                'indiv_trials_vec':         indiv_trials_vec,
                }
    
    return vec_data

def f_analyze_rd_trial_vectors(trials_rd, params, base_time = [-.250, 0], on_time = [.2, .5]):
    
    num_freq_r = len(trials_rd)
    num_freq_d = len(trials_rd[0])
    
    trial_len, num_trials, num_cells = trials_rd[0][0][0].shape
    
    plot_t1 = f_plot_t(trial_len, params['dt'])
    
    base_idx = np.logical_and(plot_t1>=base_time[0], plot_t1<base_time[1])
    on_idx = np.logical_and(plot_t1>=on_time[0], plot_t1<=on_time[1])
    
    base_mean = np.zeros((num_freq_r, num_freq_d, num_cells))
    on_mean = np.zeros((num_freq_r, num_freq_d, num_cells))
    
    base_std = np.zeros((num_freq_r, num_freq_d, num_cells))
    on_std = np.zeros((num_freq_r, num_freq_d, num_cells))
    
    base_dist_means = np.zeros((num_freq_r, num_freq_d))
    on_dist_means = np.zeros((num_freq_r, num_freq_d))
    
    base_dist_std = np.zeros((num_freq_r, num_freq_d))
    on_dist_std = np.zeros((num_freq_r, num_freq_d))
    
    indiv_mag_mean = np.zeros((num_freq_r, num_freq_d))
    indiv_mag_std = np.zeros((num_freq_r, num_freq_d))
    
    angles_mean = np.zeros((num_freq_r, num_freq_d))
    angles_std = np.zeros((num_freq_r, num_freq_d))
    
    mean_vec = np.zeros((num_freq_r, num_freq_d, num_cells))
    mean_vec_mag = np.zeros((num_freq_r, num_freq_d))
    

    indiv_trials_rd = []
    
    for n_fr in range(num_freq_r):

        indiv_trials_d = []
        for n_fd in range(num_freq_d):
            temp_data = np.concatenate(trials_rd[n_fr][n_fd], axis=1)
            
            vec_data = f_get_vec_data(temp_data, base_idx, on_idx)
            
            # save trial aves
            base_mean[n_fr, n_fd,:] = vec_data['base_mean']
            on_mean[n_fr, n_fd,:] = vec_data['on_mean']
            
            base_std[n_fr, n_fd,:] = vec_data['base_std']
            on_std[n_fr, n_fd,:] = vec_data['on_std']
            
            base_dist_means[n_fr, n_fd] = vec_data['base_dist_mean']
            on_dist_means[n_fr, n_fd] = vec_data['on_dist_mean']
            
            base_dist_std[n_fr, n_fd] = vec_data['base_dist_std']
            on_dist_std[n_fr, n_fd] = vec_data['on_dist_std']
            
            mean_vec[n_fr, n_fd,:] = vec_data['mean_vec']
            mean_vec_mag[n_fr, n_fd] = vec_data['mean_vec_mag']
            
            indiv_mag_mean[n_fr, n_fd] = vec_data['indiv_trial_mag_mean']
            indiv_mag_std[n_fr, n_fd] = vec_data['indiv_trial_mag_std']
            
            angles_mean[n_fr, n_fd] = vec_data['indiv_trial_angles_mean']
            angles_std[n_fr, n_fd] = vec_data['indiv_trial_angles_std']
            
            indiv_trials_d.append(vec_data['indiv_trials_vec'])
        
        indiv_trials_rd.append(indiv_trials_d)
        
    

    data_out = {'base_mean':                base_mean,
                'on_mean':                  on_mean,
                'base_dist_mean':           base_dist_means,
                'base_dist_std':            base_dist_std,
                'on_dist_mean':             on_dist_means,
                'on_dist_std':              on_dist_std,
                'mean_vec':                 mean_vec,
                'mean_vec_mag':             mean_vec_mag,
                'indiv_trial_mag_mean':     indiv_mag_mean,
                'indiv_trial_mag_std':      indiv_mag_std,
                'indiv_trial_angles_mean':  angles_mean,
                'indiv_trial_angles_std':   angles_std,
                'indiv_trials':             indiv_trials_rd,
                }

    return data_out


def f_analyze_cont_trial_vectors(trials_all, params, base_time = [-.250, 0], on_time = [.2, .5]):
    
    num_freqs = len(trials_all)
    trial_len, _, num_cells = trials_all[0].shape
    
    plot_t1 = f_plot_t(trial_len, params['dt'])
    
    base_idx = np.logical_and(plot_t1>base_time[0], plot_t1<base_time[1])
    on_idx = np.logical_and(plot_t1>on_time[0], plot_t1<on_time[1])
    
    base_mean = np.zeros((num_freqs, num_cells))
    on_mean = np.zeros((num_freqs, num_cells))
    
    base_std = np.zeros((num_freqs, num_cells))
    on_std = np.zeros((num_freqs, num_cells))
    
    base_dist_mean = np.zeros((num_freqs))
    on_dist_mean = np.zeros((num_freqs))
    
    base_dist_std = np.zeros((num_freqs))
    on_dist_std = np.zeros((num_freqs))
    
    indiv_mag_mean = np.zeros((num_freqs))
    indiv_mag_std = np.zeros((num_freqs))
    
    mean_vec = np.zeros((num_freqs, num_cells))
    mean_vec_mag = np.zeros((num_freqs))
    
    angles_mean = np.zeros((num_freqs))
    angles_std = np.zeros((num_freqs))
    
    indiv_trials = []
    
    for n_freq in range(num_freqs):
        temp_data = trials_all[n_freq]
        
        vec_data = f_get_vec_data(temp_data, base_idx, on_idx)
        
        # save stuff
        base_mean[n_freq,:] = vec_data['base_mean']
        on_mean[n_freq,:] = vec_data['on_mean']
        
        base_std[n_freq,:] = vec_data['base_std']
        on_std[n_freq,:] = vec_data['on_std']
        
        base_dist_mean[n_freq] = vec_data['base_dist_mean']
        on_dist_mean[n_freq] = vec_data['on_dist_mean']
        
        base_dist_std[n_freq] = vec_data['base_dist_std']
        on_dist_std[n_freq] = vec_data['on_dist_std']
        
        
        mean_vec[n_freq,:] = vec_data['mean_vec']
        mean_vec_mag[n_freq] = vec_data['mean_vec_mag']
        
        indiv_mag_mean[n_freq] = vec_data['indiv_trial_mag_mean']
        indiv_mag_std[n_freq] = vec_data['indiv_trial_mag_std']
        
        angles_mean[n_freq] = vec_data['indiv_trial_angles_mean']
        angles_std[n_freq] = vec_data['indiv_trial_angles_std']
        
        indiv_trials.append(vec_data['indiv_trials_vec'])
    
    data_out = {'base_mean':                base_mean,
                'on_mean':                  on_mean,
                'base_dist_mean':           base_dist_mean,
                'base_dist_std':            base_dist_std,
                'on_dist_mean':             on_dist_mean,
                'on_dist_std':              on_dist_std,
                'mean_vec':                 mean_vec,
                'mean_vec_mag':             mean_vec_mag,
                'indiv_trial_mag_mean':     indiv_mag_mean,
                'indiv_trial_mag_std':      indiv_mag_std,
                'indiv_trial_angles_mean':  angles_mean,
                'indiv_trial_angles_std':   angles_std,
                'indiv_trials':             indiv_trials,
                }
    
    return data_out

# old version with runs separated
# def f_analyze_cont_trial_vectors(rates_cont4d, trials_cont_cut, freqs_list, params, base_time = [-.250, 0], on_time = [.2, .5]):
    
#     trial_len, num_trials, num_runs, num_cells = rates_cont4d.shape
    
#     plot_t1 = f_plot_t(trial_len, params['dt'])
    
#     base_idx = np.logical_and(plot_t1>base_time[0], plot_t1<base_time[1])
#     on_idx = np.logical_and(plot_t1>on_time[0], plot_t1<on_time[1])
    
#     num_trials, num_runs = trials_cont_cut.shape
    
#     freqs_all = np.unique(freqs_list)
#     num_freqs = freqs_all.shape[0]
    
#     trials_all = []
    
#     for n_freq in range(num_freqs):
#         freq1 = freqs_all[n_freq]
#         trials_all2 = []
#         for n_run in range(num_runs):
#             tr_idx = trials_cont_cut[:,n_run] == freq1
        
#             temp_data = rates_cont4d[:,tr_idx,n_run,:]
            
#             trials_all2.append(temp_data)
            
#         trials_all.append(trials_all2)
    
#     base_mean = np.zeros((num_freqs, num_runs, num_cells))
#     on_mean = np.zeros((num_freqs, num_runs, num_cells))
    
#     base_std = np.zeros((num_freqs, num_runs, num_cells))
#     on_std = np.zeros((num_freqs, num_runs, num_cells))
    
#     base_dist_mean = np.zeros((num_freqs, num_runs))
#     on_dist_mean = np.zeros((num_freqs, num_runs))
    
#     mean_indiv_mag = np.zeros((num_freqs, num_runs))
    
#     mean_mag = np.zeros((num_freqs, num_runs))
    
#     mean_angles = np.zeros((num_freqs, num_runs))
    
#     mean_vec_dir = np.zeros((num_freqs, num_runs, num_cells))
    
#     for n_freq in range(num_freqs):
#         for n_run in range(num_runs):
#             temp_data = trials_all[n_freq][n_run]
            
#             temp_base = np.mean(temp_data[base_idx,:,:], axis=0)
#             temp_on = np.mean(temp_data[on_idx,:,:], axis=0)
            
#             temp_base_mean = np.mean(temp_base, axis=0)
#             temp_on_mean = np.mean(temp_on, axis=0)
            
#             base_std[n_freq, n_run,:] = np.std(temp_base, axis=0)
#             on_std[n_freq, n_run,:] = np.std(temp_on, axis=0)
            
#             base_mean[n_freq, n_run,:] = temp_base_mean
#             on_mean[n_freq, n_run,:] = temp_on_mean
            
#             mean_vec_dir[n_freq, n_run,:] = temp_on_mean - temp_base_mean
            
#             base_dist_mean[n_freq, n_run] = np.mean(f_euc_dist(temp_base, temp_base_mean))
#             on_dist_mean[n_freq, n_run] = np.mean(f_euc_dist(temp_on, temp_on_mean))
            
#             mean_indiv_mag[n_freq, n_run] = np.mean(f_euc_dist(temp_base, temp_on))
            
#             mean_mag[n_freq, n_run] = f_euc_dist(temp_base_mean, temp_on_mean)
            
#             trial_directions = temp_on - temp_base
#             mean_direction = np.mean(trial_directions, axis=0)
    
#             vec_angles = f_cos_sim(trial_directions.T, mean_direction)
#             #vec_angles2 = 1 - pdist(np.vstack((mean_direction, trial_directions)), 'cosine')
            
#             mean_angles[n_freq, n_run] = np.mean(vec_angles)
    
#     data_out = {'base_dist_mean':       base_dist_mean,
#                 'on_dist_mean':         on_dist_mean,
#                 'mean_indiv_mag':       mean_indiv_mag,
#                 'mean_mag':             mean_mag,
#                 'mean_angles':          mean_angles,
#                 'mean_vec_dir':         mean_vec_dir}
    
#     return data_out

#%%

def f_get_trace_tau(trace, sm_bin = 0):
    
    #sm_bin = 10#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    
    
    tracen = trace - np.mean(trace)
    tracen = tracen/np.std(tracen)
    
    corr1 = correlate(tracen, tracen)/len(tracen)
    
    #lags = correlation_lags(len(tracen), len(tracen))
    
    if sm_bin:
        kernel = np.ones(sm_bin)/sm_bin
        corr1_sm = np.convolve(corr1, kernel, mode='same')
        
        corr1_smn = corr1_sm - np.mean(corr1_sm)
        corr1_smn = corr1_smn/np.max(corr1_smn)
    else:
        corr1_smn = corr1
    

    
    corr1_smn2 = corr1_smn[len(trace)-1:]
    
    # plt.figure(); plt.plot(corr1_smn2)
    
    tau_corr = np.where(corr1_smn2 < 0.5)[0][0]
    
    # x = np.arange(corr_len)+1
    # y = corr1[num_trials2*num_run:num_trials2*num_run+corr_len]
    
    # yn = y - np.min(y)+0.01
    # yn = yn/np.max(yn)
    
    # fit = np.polyfit(x, np.log(yn), 1)  
    
    # y_fit = np.exp(x*fit[0]+fit[1])
    
    # tau_corr = np.log(1/2)/fit[0]*params['dt']
    
    # x = np.random.rand(1000)
    # corrx = correlate(x, x)
    # plt.figure(); plt.plot(corrx)
    
    return tau_corr, corr1_smn2

def f_get_tr_ave_tau(trace, sm_bin = 10):
    
    #sm_bin = 10#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    
    
    tracen = trace - np.mean(trace)
    tracen = tracen/np.std(tracen)
    
    corr1 = correlate(tracen, tracen)
    
    #lags = correlation_lags(len(tracen), len(tracen))
    
    if sm_bin:
        kernel = np.ones(sm_bin)/sm_bin
        corr1_sm = np.convolve(corr1, kernel, mode='same')
    else:
        corr1_sm = corr1
    
    corr1_smn = corr1_sm - np.min(corr1_sm)
    corr1_smn = corr1_smn/np.max(corr1_smn)
    
    corr1_smn2 = corr1_smn[len(trace)-1:]
    
    # plt.figure(); plt.plot(corr1_smn2)
    
    tau_corr = np.where(corr1_smn2 < 0.5)[0][0]
    
    # x = np.arange(corr_len)+1
    # y = corr1[num_trials2*num_run:num_trials2*num_run+corr_len]
    
    # yn = y - np.min(y)+0.01
    # yn = yn/np.max(yn)
    
    # fit = np.polyfit(x, np.log(yn), 1)  
    
    # y_fit = np.exp(x*fit[0]+fit[1])
    
    # tau_corr = np.log(1/2)/fit[0]*params['dt']
    
    # x = np.random.rand(1000)
    # corrx = correlate(x, x)
    # plt.figure(); plt.plot(corrx)
    
    return tau_corr, corr1_smn2

#%% vector magnitudes and angles

def f_get_rd_vec_mags(indiv_trials_in):
    num_red = len(indiv_trials_in)
    num_dev = len(indiv_trials_in[0])
    num_cells = indiv_trials_in[0][0].shape[-1]
    
    # collect data
    indiv_trials_all = []
    resp_mean_vec_rd = np.zeros((num_red, num_dev, num_cells))
    
    for n_r in range(num_red):
        for n_d in range(num_dev):
            indiv_vec_temp = indiv_trials_in[n_r][n_d]
            indiv_trials_all.append(indiv_vec_temp)
            
            resp_mean_vec_rd[n_r,n_d,:] = np.mean(indiv_vec_temp, axis=0)
     
    # individual magnitudes
    indiv_trials_all2 = np.concatenate(indiv_trials_all, axis=0) 
    indiv_trials_mag_all = np.sqrt(np.sum(indiv_trials_all2**2, axis=1))
    indiv_mags = np.mean(indiv_trials_mag_all)
    
    # red dev specific magnitudes
    resp_mean_vec_rd2d = np.reshape(resp_mean_vec_rd, (num_red*num_dev, num_cells), order='F')
    resp_mean_vec_rd_mag = np.sqrt(np.sum(resp_mean_vec_rd2d**2, axis=1))
    mean_mag_rd = np.mean(resp_mean_vec_rd_mag)
    
    # deviant freq specific mag, combine over red
    resp_mean_vec_d = np.mean(resp_mean_vec_rd, axis=0)
    resp_mean_vec_d_mag = np.sqrt(np.sum(resp_mean_vec_d**2, axis=1))
    mean_mag_d = np.mean(resp_mean_vec_d_mag)
    
    # combine over all red and dd freqs
    resp_vec_mean_all = np.mean(resp_mean_vec_d, axis=0)
    mean_mag_all = np.sqrt(np.sum(resp_vec_mean_all**2, axis=0))

    return indiv_mags, mean_mag_rd, mean_mag_d, mean_mag_all


def f_get_cont_vec_mags(indiv_trials_in):
    num_cont = len(indiv_trials_in)
    num_cells = indiv_trials_in[0].shape[-1]
    
    # collect data
    indiv_trials_all = []
    resp_mean_vec = np.zeros((num_cont, num_cells))
    
    for n_c in range(num_cont):
        indiv_vec_temp = indiv_trials_in[n_c]
        indiv_trials_all.append(indiv_vec_temp)
        
        resp_mean_vec[n_c,:] = np.mean(indiv_vec_temp, axis=0)
     
    # individual magnitudes
    indiv_trials_all2 = np.concatenate(indiv_trials_all, axis=0) 
    indiv_trials_mag_all = np.sqrt(np.sum(indiv_trials_all2**2, axis=1))
    indiv_mags = np.mean(indiv_trials_mag_all)
    
    # freq specific mag, combine over red
    resp_mean_vec_f = np.mean(resp_mean_vec, axis=0)
    resp_mean_vec_f_mag = np.sqrt(np.sum(resp_mean_vec_f**2))
    mean_mag_f = np.mean(resp_mean_vec_f_mag)
    
    # combine over all red and dd freqs
    resp_vec_mean_all = np.mean(resp_mean_vec_f, axis=0)
    mean_mag_all = np.sqrt(np.sum(resp_vec_mean_all**2, axis=0))

    return indiv_mags, mean_mag_f, mean_mag_all

def f_get_rd_vec_mags2(indiv_trials_rd):
    num_red = len(indiv_trials_rd)
    num_dev = len(indiv_trials_rd[0])
    num_cells = indiv_trials_rd[0][0].shape[-1]
    
    # collect data
    
    resp_mean_vec_rd = np.zeros((num_red, num_dev, num_cells))
    
    mean_rd_mags = np.zeros((num_red, num_dev))
    mean_rd_cossim = np.zeros((num_red, num_dev))
    
    indiv_trials_all = []
    indiv_trials_dr = []
    
    for n_d in range(num_dev):
        indiv_trials_dr2 = []
        for n_r in range(num_red):
            
            indiv_vec_temp = indiv_trials_rd[n_r][n_d]
            indiv_trials_dr2.append(indiv_vec_temp)
            
            indiv_trials_all.append(indiv_vec_temp)
            
            trial_ave_rd_vec = np.mean(indiv_vec_temp, axis=0)[None,:]
            resp_mean_vec_rd[n_r,n_d,:] = trial_ave_rd_vec
            
            # deviant + redundant specific
            dev_red_resp_mag = np.sqrt(np.sum(trial_ave_rd_vec**2, axis=1))
            mean_rd_mags[n_r,n_d] = dev_red_resp_mag[0]
            
            cs = 1-pdist(indiv_vec_temp, 'cosine')
            mean_rd_cossim[n_r,n_d] = np.mean(cs)
            
        indiv_trials_dr.append(indiv_trials_dr2)
    
    mean_rd_mags2 = np.mean(mean_rd_mags)
    mean_rd_cossim2 = np.mean(mean_rd_cossim)
    
    # individual magnitudes isolated; cos similarity is 1
    indiv_trials_all2 = np.concatenate(indiv_trials_all, axis=0) 
    indiv_trials_mag_all = np.sqrt(np.sum(indiv_trials_all2**2, axis=1))
    mean_indiv_mags = np.mean(indiv_trials_mag_all)
    
    # dev specific magnitudes, combine over red
    mean_dev_mags = np.zeros((num_dev))
    mean_dev_cossim = np.zeros((num_dev))
    for n_d in range(num_dev):
        dev_resp1 = np.concatenate(indiv_trials_dr[n_d], axis=0)
        
        trial_ave_dev_vec = np.mean(dev_resp1, axis=0)[None,:]
        
        dev_resp1_mag = np.sqrt(np.sum(trial_ave_dev_vec**2, axis=1))
        mean_dev_mags[n_d] = dev_resp1_mag[0]
        
        cs = 1-pdist(dev_resp1, 'cosine')
        mean_dev_cossim[n_d] = np.mean(cs)
    
    mean_dev_mags2 = np.mean(mean_dev_mags)
    mean_dev_cossim2 = np.mean(mean_dev_cossim)
    
    # now combined across red and dev
    total_vec = np.mean(indiv_trials_all2, axis=0)
    mean_total_vec_mag = np.sqrt(np.sum(total_vec**2))
    
    total_cossim = 1-pdist(indiv_trials_all2, 'cosine')
    mean_total_cossim = np.mean(total_cossim)
    
    return mean_indiv_mags, mean_rd_mags2, mean_dev_mags2, mean_total_vec_mag, mean_rd_cossim2, mean_dev_cossim2, mean_total_cossim


def f_get_cont_vec_mags2(indiv_trials_c):
    num_cont = len(indiv_trials_c)
    num_cells = indiv_trials_c[0].shape[-1]
    
    # collect data
    indiv_trials_all = []
    resp_mean_vec = np.zeros((num_cont, num_cells))
    
    mean_c_mags = np.zeros((num_cont))
    mean_c_cossim = np.zeros((num_cont))
    
    for n_c in range(num_cont):
        indiv_vec_temp = indiv_trials_c[n_c]
        indiv_trials_all.append(indiv_vec_temp)
        
        trial_ave_vec = np.mean(indiv_vec_temp, axis=0)[None,:]
        resp_mean_vec[n_c,:] = trial_ave_vec
        
        trial_ave_mag = np.sqrt(np.sum(trial_ave_vec**2, axis=1))
        mean_c_mags[n_c] = trial_ave_mag[0]
        
        cs = 1-pdist(indiv_vec_temp, 'cosine')
        mean_c_cossim[n_c] = np.mean(cs)
        
    mean_c_mags2 = np.mean(mean_c_mags)
    mean_c_cossim2 = np.mean(mean_c_cossim)
        
    # individual magnitudes
    indiv_trials_all2 = np.concatenate(indiv_trials_all, axis=0) 
    indiv_trials_mag_all = np.sqrt(np.sum(indiv_trials_all2**2, axis=1))
    mean_indiv_mags = np.mean(indiv_trials_mag_all)
    
    total_vec = np.mean(indiv_trials_all2, axis=0)[None,:]
    total_vec_mag = np.sqrt(np.sum(total_vec**2, axis=1))[0]
    
    total_cossim = 1-pdist(indiv_trials_all2, 'cosine')
    mean_total_cossim = np.mean(total_cossim)
    
    return mean_indiv_mags, mean_c_mags2, total_vec_mag, mean_c_cossim2, mean_total_cossim
  