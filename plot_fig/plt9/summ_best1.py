import pdb


def _summ_save_seq_o_zxg(save_seq_o):
    '''
    input: [], each line [nd, lev, actList]
    '''

    _, _, idx1 = min([[val[0], val[1], idx] for idx, val in enumerate(save_seq_o)]) # best and in this episode
    _, _, idx2 = min([[val[1], val[0], idx] for idx, val in enumerate(save_seq_o)]) # best lev in this episode

    val1 = save_seq_o[idx1]
    val2 = save_seq_o[idx2]
    print_cmd1 = ';'.join(val1[2])
    print_cmd2 = ';'.join(val2[2])
    print('best save:\n   best and: step %d, line %d, [%.1f %.1f]  %s'%(len(val1[2]), idx1, val1[0], val1[1], print_cmd1))
    print('best save:\n   best lev: step %d, line %d, [%.1f %.1f]  %s'%(len(val2[2]), idx2, val2[0], val2[1], print_cmd2))

