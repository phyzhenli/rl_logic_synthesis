
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pdb

from collections import deque

'''compare n_env=1, en_env=4'''

def read_txt(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
    return lines


def parse_sb3_zxg_csv(path6, len_deque):
    '''return 
        list, len=num_epoch'''
    lines = read_txt(path6)
    
    assert (lines[1].strip()=='r,l,t,nd,lev,seq') or (lines[1].strip()=='r,l,t,nd,lev,seq,step_q')
    r_list = []
    nd_list = []
    lev_list = []
    actionlist_list = []
    for line in lines[2:]:
        line = line.strip()

        r_ = float(line.split(',')[0])
        len_ = int(line.split(',')[1])
        nd = float(line.split(',')[3])
        lev = float(line.split(',')[4])
        actions = line.split(',')[5].split('; ')  # * Note! split('; ')

        r_list.append(r_)
        nd_list.append(nd)
        lev_list.append(lev)
        actionlist_list.append(actions)

    avg_r_list = _calc_avg_deque(r_list, len_deque)

    return avg_r_list, nd_list, lev_list, actionlist_list

def parse_sb3_step_metric(path6):
    ''' plt时，判断each step '''
    lines = read_txt(path6)
    if lines[1].strip()!='r,l,t,nd,lev,seq,step_q':
        print('no step_q')
        return None
    
    step_q_list = []
    for line in lines[2:]:
        len_ = int(line.split(',')[1])
        str_ = line.split(',')[6].split('(')
        step_q = []
        for item in str_:
            if ')' not in item: 
                continue
            str1_ = item.split(')')[0]
            area_, delay_ = re.split(r"[ ]+", str1_)
            area_, delay_ = float(area_), float(delay_)
            step_q.append([area_, delay_])
        assert len(step_q) == len_
        step_q_list.append(step_q)
    return step_q_list


def _calc_avg_deque(r_list, len_deque):
    '''calc avg deque100
        return list, len=epoch'''
    scores_window = deque(maxlen=len_deque)  
    avg_r_list = []
    for current_ep_reward in r_list:
        scores_window.append(current_ep_reward)
        avg_r = np.mean(scores_window)
        avg_r_list.append(avg_r)
    return avg_r_list


def parse_sb3_zxg_csv_multi(path7, len_deque=100):
    '''parse from n_env
        return dict
        each list, shape= n_ev * epoch '''
    assert path7.endswith('/')

    save_avgs = []
    save_nds  = []
    save_levs = []
    save_acts = []
    save_step_q = []
    for i_ in range(0, 100):
        path = path7 +str(i_) +'.monitor.csv'
        if not os.path.exists(path):
            break
        avg_r_list_, nd_list, lev_list, actionlist_list = parse_sb3_zxg_csv(path, len_deque=len_deque)
        save_avgs.append(avg_r_list_)
        save_nds.append(nd_list) 
        save_levs.append(lev_list)
        save_acts.append(actionlist_list)

        step_q_list = parse_sb3_step_metric(path)
        save_step_q.append(step_q_list)
        has_step_q = (step_q_list!=None)

    # print n_env
    abspath = '/'.join(path7.split('/')[-2:])
    print('\n', abspath,', n_env:', len(save_avgs))
    
    result = {
        'save_avgs': save_avgs,
        'save_nds': save_nds,
        'save_levs': save_levs,
        'save_acts': save_acts,
        'save_step_q': save_step_q,
        'has_step_q': has_step_q,
        }
    return result

def avg_over_n_env(result):
    '''average over n_env 
        return dict'''
    save_avgs = result['save_avgs']
    save_nds  = result['save_nds']
    save_levs = result['save_levs']
    save_acts = result['save_acts']
    save_step_q = result['save_step_q']

    # watch length
    watch_lens = [len(ll) for ll in save_avgs]
    if len(set(watch_lens)) !=1 :
        print('Note: proc 1 ~proc n, len not equal ')
        watch_length = min(watch_lens)
    else:
        watch_length = watch_lens[0]
    save_avgs = np.array( [ll[0:watch_length] for ll in save_avgs] )
    save_nds  = np.array( [ll[0:watch_length] for ll in save_nds] )
    save_levs = np.array( [ll[0:watch_length] for ll in save_levs] )
    if result['has_step_q']: save_step_q = np.array( [ll[0:watch_length] for ll in save_step_q] )

    # average over    
    avgs_env4 = np.mean(save_avgs, axis=0)
    nds_env4  = np.mean(save_nds, axis=0)
    levs_env4  = np.mean(save_levs, axis=0)

    # return dict
    result['save_avgs'] = save_avgs # len= n_env * epoch
    result['save_nds']  = save_nds  
    result['save_levs'] = save_levs 
    result['save_acts'] = save_acts 
    result['save_step_q'] = save_step_q 
    #
    result['avgs_env4'] = avgs_env4 # len= epoch
    result['nds_env4']  = nds_env4  
    result['levs_env4'] = levs_env4 
    return result

def run2():
    path1 = '/home/yuqian/Documents/2021_07_14_A/synthesis_workspace/for_lizhen/2023-04-25/logs_scl1/ppo/abc-asic-v0_176/'      

    result1 = parse_sb3_zxg_csv_multi(path1)
    result1 = avg_over_n_env(result1)

    fig_dir = path1

    fig1 = plt.figure()
    
    plt.xlabel('epoch')
    plt.ylabel('avg sum reward (deque100)')
    plt.title('')

    def get_axis(a, b):
        epochs = list(np.arange(a, b))
        return epochs

    plt.plot(get_axis(0, len(result1['avgs_env4'])), result1['avgs_env4'], linewidth=1.5, linestyle='--')

    plt.show()
    fig1.savefig(fig_dir+'avg-r.png')

    # --------------------------
    fig2 = plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('node')
    plt.title('')

    plt.plot(get_axis(0, len(result1['nds_env4'])), result1['nds_env4'], linewidth=1.5, linestyle='--')

    plt.show()
    fig2.savefig(fig_dir+'nd.png')

    # ------------------------
    fig3 = plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('lev')
    plt.title('')

    plt.plot(get_axis(0, len(result1['levs_env4'])), result1['levs_env4'], linewidth=1.5, linestyle='--')

    plt.show()
    fig3.savefig(fig_dir+'lev.png')

    # ------------------------
    from summ_best1 import _summ_save_seq_o_zxg
    n_env = len(result1['save_avgs'])
    len_epoch = len(result1['avgs_env4'])
    clamp = False
    if clamp:
        len_epoch = 1800
        len_epoch = int(np.ceil(len_epoch /n_env))
    print('clamp:', clamp)
    save_seq_o_all = []
    for idx in range(0, n_env):
        print('\n\nenv-, ', idx)
        save_seq_o = []

        # make save_seq_o
        for jdx in range(0, len_epoch):
            ll_ = [None]*3
            ll_[0] = result1['save_nds'][idx][jdx]
            ll_[1] = result1['save_levs'][idx][jdx]
            ll_[2] = result1['save_acts'][idx][jdx]
            save_seq_o.append(ll_)
            save_seq_o_all.append(ll_)  

        _summ_save_seq_o_zxg(save_seq_o)
    print('\n\nall n epoch ------')
    _summ_save_seq_o_zxg(save_seq_o_all)
    print('-------------\n')

    
    # manually filter
    summ_filtered1 = [line for line in save_seq_o_all 
                                                        # if len(line[2])<=20 

                                                        # if line[1]<=800    # lev
                                                        # if line[0]<= 800  # nd
                                                        # if line[1]>=1000
                                                    ]
    print('\n\nall n epoch, filter1, ------')
    _summ_save_seq_o_zxg(summ_filtered1)
    print('-------------\n')


    # summ adp
    from summ_quality1 import _summ_save_seq_o_zxg_adp
    _summ_save_seq_o_zxg_adp(save_seq_o_all)
    print('-------------\n')

    # summ step_q. 2022-10-07
    from summ_quality1 import _summ_save_step_quality
    if result1['has_step_q']:
        print('\nwe has step q')
        _summ_save_step_quality(result1['save_step_q'], result1['save_acts'])
    print('-------------\n')

    return 

run2()