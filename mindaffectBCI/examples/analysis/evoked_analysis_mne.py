import mne
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test

import numpy as np
import matplotlib.pyplot as plt
from mindaffectBCI.decoder.offline.load_mindaffectBCI import load_mindaffectBCI_raw_mne, make_onset_offset_events
from mindaffectBCI.decoder.utils import askloadsavefile
from mindaffectBCI.examples.analysis.listening_effort_induced_analysis_mne import load_preprocess_and_epoch, average_per_condition, cluster_test_and_plot_condition_pairs, make_cross_session_epochs, decorrelate_artifact_channels, mark_bad_channels

import os
from glob import glob

def visualize_conditions_erp(condition_ave, sessdir, label, vmin=None, vmax=None, topo_times='peaks'):
    # 7) Grand-average erp plot
    print("Plot grand average")
    grand_ave = mne.grand_average([c for c in condition_ave.values()])
    if vmin is None:
        vmin = np.min(np.percentile(grand_ave.data,10,axis=-1))
    if vmax is None:
        vmax = np.max(np.percentile(grand_ave.data,90,axis=-1))

    # TODO[]: work out how to use a consistent scale!!
    grand_ave.plot_joint(exclude='bads', title='grand average\n{}'.format(label), 
                        times=topo_times, show=False);    
    plt.savefig(os.path.join(sessdir,'grand_average_erp_{}.png'.format(label)))

    grand_ave.plot_topo(title='{}\n{}'.format('grand_ave',label), show=False)
    plt.savefig(os.path.join(sessdir,'{}_erp_topo_{}.png'.format('grand_ave',label)))

    # 8) ERP plot per condition
    for lab, ave in condition_ave.items():
        ave.plot_joint(title ='ERP {}\n{}'.format(lab,label), exclude='bads', 
                     times=topo_times, 
                     show=False)
        plt.savefig(os.path.join(sessdir,'{}_erp_{}.png'.format(lab.replace('/','_'),label)))

        ave.plot_topo(title='{}\n{}'.format(lab,label), show=False)

    mne.viz.plot_evoked_topo([d for d in condition_ave.values()],show=False)
    plt.suptitle("{}".format(label))
    plt.savefig(os.path.join(sessdir,'{}_erp_topo_{}.png'.format('all_conditions',label)))

def cluster_test_and_plot_condition_pairs_erp(epochs,condition_pairs,sessdir,label,data_type:str='erp',p_value_threshold=.1,
                cluster_params=dict(n_jobs=8,n_permutations=1000,tail=0)):#threshold=dict(start=.2, step=.2),
    print("Signficance testing...")

    for c1,c2 in condition_pairs:
        contrast = "{} vs {}".format(c1,c2)
        
        c1_epochs = epochs[c1].load_data().pick_types(eeg=True).get_data().transpose(0, 2, 1)
        c2_epochs = epochs[c2].load_data().pick_types(eeg=True).get_data().transpose(0, 2, 1)
        
        adjacency, _ = mne.channels.find_ch_adjacency(epochs[c1].load_data().pick_types(eeg=True).info, "eeg")

        T_obs, clusters, cluster_p_values, H0 = \
            spatio_temporal_cluster_test([c1_epochs, c2_epochs],
            adjacency=adjacency,**cluster_params)

        # find significatn points    
        significant_points = cluster_p_values.reshape(T_obs.shape).T < p_value_threshold
        
        # We need an evoked object to plot the image to be masked
        evoked = mne.combine_evoked([epochs[c1].load_data().pick_types(eeg=True).average(), epochs[c2].load_data().pick_types(eeg=True).average()],
                                    weights=[1, -1])  # calculate difference wave
        time_unit = dict(time_unit="s")
        

        # Create ROIs by checking channel labels
        selections = mne.channels.make_1020_channel_selections(evoked.info, midline="12z")

        # Visualize the results
        evoked.plot_joint(title="{}\n{}".format(contrast,label), show=False)  # show difference wave
        plt.savefig(os.path.join(sessdir,'{}_erp_contrast_{}.png'.format(contrast,label)))
        fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
        axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
        evoked.plot_image(axes=axes, group_by=selections, colorbar=False, show=False,
                        mask=significant_points, show_names="all", titles=None,
                        **time_unit)
        plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=.3,
                    label="µV")
        plt.title("{} (p<{})\n{}".format(contrast, p_value_threshold,label))
        plt.savefig(os.path.join(sessdir, '{}_erp_cluster_{}.png'.format(contrast, label)))


def run(exptdir:str=None, label:str='erp', filematch:str='mindaffectBCI*.txt', 
        ch_names = None, artifact_ch=['Fp2','Fp1','F9','F10'],
        trigger_type='offset',
        conditions=['onset','offset'], vmin=None, vmax=None, visualize_conditions:bool=True,
        topo_times='peaks', 
        tmin=-.1, tmax=.5, n_pre=1, n_post=1, l_freq=1, h_freq=20,
        cluster_test_condition_pairs:bool=True,
        condition_pairs=[('onset','offset')], 
        cluster_params=dict(threshold=None,n_jobs=8,n_permutations=1000,tail=0),
        cluster_p_value_threshold:float=.1
        ):
        
    # load savefile
    if exptdir is None:
        exptdir = askloadsavefile(initialdir=os.getcwd(),filetypes='dir')
        print(exptdir)

    # get the list of save files to load
    filelist = glob(os.path.join(exptdir,'**',filematch),recursive=True)
    print("Found {} matching data files\n".format(len(filelist)))
    print(filelist)

    # get the prefix to strip for short names
    commonprefix = os.path.commonprefix(filelist)

    # run each file at a time & save per-condition average response for grand-average later
    subj_condition_erp = []
    for filename in filelist:
        sessdir = os.path.dirname(filename)
        flabel = os.path.split(filename[len(commonprefix):])
        flabel = flabel[0] if flabel[0] else flabel[1]
        flabel = flabel.replace('\\','_').replace('/','_')
        flabel = flabel + '_' + label

        # location to save the figures
        savedir = os.path.join(sessdir,label)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        print("\n\n------------------------------\n{}\n---------------------\n".format(filename))
        epochs = load_preprocess_and_epoch(filename,sessdir=savedir,ch_names=ch_names, label=flabel, visualize=visualize_conditions,
                                            trigger_type=trigger_type, artifact_ch=artifact_ch, tmin=tmin, tmax=tmax, 
                                            l_freq=l_freq, h_freq=h_freq)
        plt.close('all')

        # get per-condition average response
        condition_erp_i = average_per_condition(epochs,conditions)
        subj_condition_erp.append(condition_erp_i)
        
        # plot the per-condition response 
        if visualize_conditions:
            visualize_conditions_erp(condition_erp_i, savedir, flabel, vmin=vmin, vmax=vmax, topo_times=topo_times)
        plt.close('all')

        # stats test tfr for given pairs of conditions
        if cluster_test_condition_pairs and condition_pairs is not None:
            cluster_test_and_plot_condition_pairs(epochs, condition_pairs, savedir, flabel, data_type='erp',
                                        p_value_threshold=cluster_p_value_threshold, 
                                        cluster_params=cluster_params)

        plt.close('all')
        del epochs

    # don't cross-condition if single condition
    if len(filelist)<=1:
        return

    # make a epochsTFR object with per-session + condition averages as examples
    epochs = make_cross_session_epochs(subj_condition_erp, data_type='erp')
    group_label = 'cross-session' + '_' + label
    savedir = os.path.join(exptdir,label)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # plot the cross session per-condition response and diff w.r.t. grand_average
    if visualize_conditions:
        # get per-condition average response
        condition_erp = average_per_condition(epochs,conditions)
        visualize_conditions_erp(condition_erp, savedir, group_label, vmin=vmin, vmax=vmax)

    # stats test tfr for given pairs of conditions
    if condition_pairs is not None and cluster_test_condition_pairs:
        cluster_test_and_plot_condition_pairs(epochs, condition_pairs, savedir, group_label, data_type='erp',
                                    p_value_threshold=cluster_p_value_threshold,
                                    cluster_params=cluster_params)


if __name__=='__main__':
    #exptdir = "G:/Shared drives/Data/experiments/audio_listening_effort"
    # get the experiment dir to run on
    exptdir = askloadsavefile(initialdir=os.getcwd(),filetypes='dir')
    print(exptdir)

    # channel names, BP r-NET
    ch_names = None
    ch_names=["Fp1","Fz","F3","F7","F9","FC5","FC1","C3","T7","CP5","CP1","Pz","P3","P7","P9","O1","Oz","O2","P10","P8","P4","CP2","CP6","T8","C4","Cz","FC2","FC6","F10","F8","F4","Fp2"]
    artifact_ch=['Fp2','Fp1','F9','F10']

    # look at the data in per-output basis
    conditions = ['o{:d}.l{}'.format(o,l) for o in range(1,10) for l in range(1,8)]# for e in ('onset','offset')]
    #conditions = ['o{:d}.l{}.{}'.format(o,l,e) for o in range(1,10) for l in range(1,8) for e in ('onset','offset')]
    output = 'onset_offset'

    if 1:   # in trial erp vis
        run(exptdir, visualize_conditions=True, cluster_test_condition_pairs=False, trigger_type=output,
            label='erp', conditions=conditions, ch_names=ch_names, artifact_ch=artifact_ch,
            tmin=-.1, tmax=.7, vmin=-.25, vmax=.25, l_freq=3, h_freq=20,
            cluster_params=dict(n_jobs=8,n_permutations=1000,tail=0)) #threshold=dict(start=.2, step=.2),

    if 0:   # in trial significance test
        run(exptdir, visualize_conditions=False, cluster_test_condition_pairs=True, trigger_type=output,
            label='erp_clust', conditions=conditions, ch_names=ch_names, artifact_ch=artifact_ch,
            tmin=-.1, tmax=.7, vmin=-.25, vmax=.25, l_freq=3, h_freq=20,
            cluster_params=dict(n_jobs=8,n_permutations=1000,tail=0)) #threshold=dict(start=.2, step=.2),
