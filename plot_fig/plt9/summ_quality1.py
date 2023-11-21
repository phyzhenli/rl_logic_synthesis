import pdb


def _summ_save_seq_o_zxg_adp(save_seq_o):
    '''
    input: [], each line [nd, lev, actList]
    summ area delay product 
    '''

    save_seq_o_ld = []  # list dict 
    for nd_, lev_, cmd_ in save_seq_o:
        # calc area delay product 
        adp = nd_ * lev_
        len_ = len(cmd_)
        save_seq_o_ld.append({'nd':nd_, 'lev':lev_, 'cmd':cmd_, 'adp':adp, 'len_':len_})

    # second排序 是lev 
    _, _, _, idx1 = min([[val['adp'], val['lev'], val['nd'], idx] for idx, val in enumerate(save_seq_o_ld)]) # best adp in this episode

    val1 = save_seq_o_ld[idx1]
    print_cmd1 = ';'.join(val1['cmd'])
    print('best save:\n   best adp: step %d, line %d, [%.1f %.1f], adp %f,  %s'%(
        len(val1['cmd']), idx1, val1['nd'], val1['lev'], val1['adp'], print_cmd1))


def _summ_save_step_quality(save_step_q, save_acts):
    '''save_step_q: dim [n_env, n_epoch, 2]'''
    n_env = len(save_step_q)
    len_epoch = len(save_step_q[0])
    save_q_all = []
    for idx in range(0, n_env):
        for jdx in range(0, len_epoch):  # in each epoch
            tmp_line = [[nd*lev, lev, nd, idx, jdx, kdx] for kdx, (nd, lev) in enumerate(save_step_q[idx][jdx])]
            save_q_all.extend(tmp_line)  

    p_, l_, n_, idx, jdx, kdx = min(save_q_all)
    print_cmd1 = ';'.join(save_acts[idx][jdx][0:kdx])
    print('best step_quality:')
    print('   in env-%d.csv, epoch-%d, step-%d'%(idx, jdx, kdx))
    print('   best adp (step): [%.1f %.1f], step-%d, adp %.1f,  %s'%(n_, l_, kdx, p_, print_cmd1))
