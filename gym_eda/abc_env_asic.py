from abc_py import AbcInterface
import numpy as np
import torch
import gym
import dgl
import pdb

def decode_actions(action_code):
    base, rs, extra = ['b', 'rw', 'rf'], [], []
    for c in action_code.split(';'):
        if c.startswith('rs'):
            rs = ['rs -K ' + k for k in c.split()[1:]]
        elif c[0] == 'N':
            rs.extend([rss + ' -N ' + c[1] for rss in rs])
        elif c == '-z':
            base.extend([bs[0:2] + 'z' + bs[2:] for bs in base if bs[0] == 'r'])
        elif c == '-l':
            base.extend([bs + ' -l' for bs in base])
        elif c == '+z':
            base.extend([bs[0:2] + 'z' + bs[2:] for bs in base if bs[0] == 'r'])
            rs.extend([rss[0:2] + 'z' + rss[2:] for rss in rs])
        elif c == '+l':
            base.extend([bs + ' -l' for bs in base])
            rs.extend([rss + ' -l' for rss in rs])
        elif c == 'dchb':
            base.append(c)
        else:
            extra.append(c)
    return base + rs + extra

class AbcEnv_ASIC(gym.Env):
    """ EDA environment """
    def __init__(self, bench, actions='-z;-l', optimize='mix', baseline='resyn2;resyn2', max_seq_len=20,
                 seq_end='time', runtime_reward=False, 
                 mapping='', cell_lib='', step_mapping = False, state_ver=0,
                 map_tail=False, tune_actions=None, debug_actions=False, map_latter=None, map_probe='',
                 alias_flag=None,
                 rewardD_ver=0,
                 ):
        self.abc = AbcInterface()
        self.aigfile = bench
        self.actions = decode_actions(actions)
        if tune_actions=='comp_boils':
            self.actions = ['balance', 'rewrite', 'rewrite -z', 'refactor',     
                        'refactor -z', 'resub', 'resub -z', 'strash',
                        'fraig', '&get -n;&sopb;&put;', '&get -n;&blut;&put;', '&get -n;&dsdb;&put;']
            if debug_actions:
                pass
                # self.actions.append('dc2')
                # self.actions.append('dch -f;b -l')
                # self.actions.append('&ifst')

        elif tune_actions=='add_abc9':
            self.actions.append('&get -n;&sopb;&put;')
            self.actions.append('&get -n;&blut;&put;')
            self.actions.append('&get -n;&dsdb;&put;')
            self.actions.append('fraig') # !
            if debug_actions:
                pass
                # self.actions.extend(['irw', 'drw', 'drf', 'dfraig'])
        elif tune_actions=='add_try1':
            self.actions.append('ifraig')
            # self.actions.append('dch -f;b')
            self.actions.append('dch -f;st')
            self.actions.append('if -g')
        print(self.actions)
        self.optimize = optimize
        self.baseline = baseline
        self.max_seq_len = max_seq_len
        self.seq_end = seq_end
        self.state_ver = state_ver
        self.runtime_reward = runtime_reward
        self.runTime = 1e-20
        if cell_lib: self.abc.run('r ' + cell_lib)
        self.map_mode = ''
        if mapping:  self.map_mode, self.map_arg = mapping.split(';')
        # mapping = mapping != ''
        self.map_tail=map_tail
        self.map_latter=map_latter
        self.step_mapping = step_mapping    # get mapping stats at each step instead of at each epoch
        self.map_probe = map_probe
        self.step_quality = []
        self.rewardD_ver = rewardD_ver

        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

        self.add_abc_alias(alias_flag)
        self.abc.read(self.aigfile)
        self.initStats = self.getStats()
        self.initMapStats = self.getStats(True)
        self.getBaseTime()
        baseStats = self.getStats(self.hasMapping())
        totalReward = self.statValue(baseStats, self.baseTime, max_seq_len, self.hasMapping())
        initStats = self.initMapStats if self.hasMapping() else self.initStats
        # self.rewardBaseline = totalReward / self.max_seq_len
        print(f"\ninit {initStats} baseline {baseStats} base time {self.baseTime}  total reward {totalReward}")

    def getBaseTime(self):
        tt = 0; n = 3
        self.maxBaseTime = 0
        for i in range(n):
            self.abc.read(self.aigfile)
            t = self.abc.run(self.baseline)[0]
            self.maxBaseTime = max(self.maxBaseTime, t)
            tt += t
        self.baseTime = tt / n

    def reset(self, *, seed = None, return_info = False, options = None):
        #super().reset(seed=seed)
        self.runTime = 1e-20
        self.abc.read(self.aigfile)
        self.lastStats = self.getStats() # The initial AIG statistics
        self.curStats = self.lastStats  # the current AIG statistics
        self.actTime = [(0,0)] * self.numActions()  # average action runtimes
        self.actSeq = []                # action sequence
        self.step_quality = []
        if return_info:
            return self.state(), {}
        else:
            return self.state()

    def hasMapping(self):
        return self.map_mode != ''

    def numActions(self):
        return len(self.actions)

    def lastAction(self, i):
        return self.actSeq[-i] if i < len(self.actSeq) else -1

    def getStats(self, run_mapping = False):
        # if self.step_mapping: run_mapping = True
        if self.step_mapping:  # * then always probe 
            assert self.map_mode == 'SCL'
            # pdb.set_trace() 
            if self.map_probe == 'drills_probe':
                str_ = 'backup;ifraig;map;topo;'
                if self.map_arg!='':  str_ = ('map -D %s'%(self.map_arg)).join(str_.split("map"))
                self.abc.run(str_)
                # print(self.abc.sclStats()) 
                ret = self.abc.sclStats()
                self.abc.run('restore')
            
            if run_mapping==False: # if it is not last step
                return ret + (-1,)
        if run_mapping and self.map_mode == 'SCL':
            if not self.map_tail:  # False
                assert self.map_arg!=''
                self.abc.runcmd('map', D=self.map_arg)
                self.abc.run('topo')
            elif self.map_tail == 'flowtune':
                # str_ = 'strash;ifraig;dch -f;map;topo;upsize;dnsize;topo;'       # flowtune
                str_ = 'strash;ifraig;map;topo;upsize;dnsize;topo;'             # remove dch -f
                # str_ = 'strash;ifraig;dch -f;strash;map;topo;upsize;dnsize;topo;' # with dch -f, add st 
                # str_ = 'strash;balance;ifraig;map;topo;upsize;dnsize;topo;'             # remove dch -f, add balance
                # str_ = 'strash;ifraig;map;mfs2;topo;upsize;dnsize;topo;'             # remove dch -f, add mfs2. error. cannot execute 
                if self.map_arg!='':  str_ = ('map -D %s'%(self.map_arg)).join(str_.split("map"))
                self.abc.run(str_)
            elif self.map_tail == 'ref_yosys':
                # https://yosyshq.net/yosys/cmd_abc.html
                str_ = 'strash;&get -n;&dch -f;&nf;&put;buffer;upsize;dnsize;topo;'
                if self.map_arg!='':  str_ = ('&nf -D %s'%(self.map_arg)).join(str_.split("&nf"))
                self.abc.run(str_)
            elif self.map_tail == 'drills':
                str_ = 'strash;ifraig;map;topo'
                if self.map_arg!='':  str_ = ('map -D %s'%(self.map_arg)).join(str_.split("map"))
                self.abc.run(str_)
            elif self.map_tail == 'c_drills':
                str_ = 'strash;map;topo'
                if self.map_arg!='':  str_ = ('map -D %s'%(self.map_arg)).join(str_.split("map"))
                self.abc.run(str_)
            return self.abc.sclStats() + (-1,)  # area, delay

        if run_mapping and self.map_mode == 'FPGA':
            # pdb.set_trace()
            if self.map_latter=='fixed_abc9':
                self.abc.run('&get -n;&sopb;&put; if -K 6; st;')
                self.abc.run('&get -n;&blut;&put; if -K 6; st;')
                self.abc.run('&get -n;&dsdb;&put; if -K 6; st;')
            if not self.map_tail:
                self.abc.runcmd('if', K=self.map_arg)
            else:
                # self.abc.run('strash;ifraig;scorr;dc2;strash;dch -f;if -K 6;mfs2;luprun7tpack -S 1;')
                self.abc.run('strash;ifraig;scorr;dc2;strash;dch -f;if -K 6;mfs2;lutpack -S 1;')
        isaig, i, o, lat, nd, edge, lev = self.abc.ntkStats()  # The initial AIG statistics (isaig, i, o, lat, and, edge, lev)
        # pdb.set_trace()
        return nd, lev, edge

    def step(self, actionIdx):
        self.actSeq.append(actionIdx)
        self.lastTime = self.runTime
        an, at = self.actTime[actionIdx]
        t = self.abc.run(self.actions[actionIdx])[0]
        self.actTime[actionIdx] = an+1, (an*at + t)/(an+1)
        self.runTime += self.actTime[actionIdx][1]
        # done = self.runTime > self.maxBaseTime if self.seq_end == 'time' else len(self.actSeq) >= self.max_seq_len
        done = len(self.actSeq) >= self.max_seq_len or self.seq_end == 'time' and self.runTime > self.maxBaseTime
        # update the statitics
        self.lastStats = self.curStats
        self.curStats = self.getStats(done)
        nextState = self.state()
        reward = self.reward(done)
        cmdSeq = '; '.join([self.actions[id] for id in self.actSeq])
        self.step_quality.append((float(format(self.curStats[0], '.1f')), float(format(self.curStats[1], '.1f')))) # (area, delay) # write to file 
        # pdb.set_trace()
        return nextState, reward, done, {'nd':self.curStats[0], 'lev':self.curStats[1], 'seq':cmdSeq, 'step_q':str(self.step_quality).replace(',', '')}
    

    def state(self):
        nHist = 3
        stateArray = np.array([self.curStats[0] / self.initStats[0], self.curStats[1] / self.initStats[1],
                               self.lastStats[0] / self.initStats[0], self.lastStats[1] / self.initStats[1]])
        lastOneHotActs  = np.zeros(self.numActions() + 1)
        for i in range(1, nHist + 1):
            lastOneHotActs[self.lastAction(i)] += 1/nHist
        stepArray = np.array([self.runTime / self.baseTime, len(self.actSeq) / self.max_seq_len])
        if   self.state_ver == 0: result = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        elif self.state_ver == 1: result = np.concatenate((stateArray, lastOneHotActs), axis=-1)
        elif self.state_ver == 4: result = np.concatenate((lastOneHotActs, stepArray), axis=-1)
        elif self.state_ver == 2: result = np.concatenate((stateArray, stepArray), axis=-1)
        elif self.state_ver == 3: result = stateArray
        return result.astype(np.float32)

    def dimState(self):
        lenState, lenOneHot, lenStep = 4, self.numActions() + 1, 2
        return (lenState+lenOneHot+lenStep, lenState+lenOneHot, lenState+lenStep, lenState, lenOneHot+lenStep)[self.state_ver]

    def reward(self, done):
        curVal  = self.statValue(self.curStats, self.runTime, len(self.actSeq), done)
        lastVal = self.statValue(self.lastStats, self.lastTime, len(self.actSeq) - 1)
        # base = 0 if self.seq_end == 'time' else self.rewardBaseline

        # print(curVal - lastVal)
        # if done: pdb.set_trace()
        # if done==True: return curVal # * qy
        r_ = curVal - lastVal
        if self.rewardD_ver != 0:
            assert self.map_arg != '' 
            D_c = int(self.map_arg) # constr 
            D_t = self.curStats[1]
        if self.rewardD_ver == 1:
            if done:  # only when last step 
                # pdb.set_trace()
                r_ = curVal - lastVal
                beta = 1.0 
                r_ = r_ *beta

                if r_ >= 0:
                    if D_t < D_c:
                        r_ = r_ *1
                    elif D_t >1.2*D_c:
                        r_ = 0
                    else:
                        r_ = r_*( 1+(D_c-D_t)/D_c  )   # [0.8r, r]
                if r_ < 0: 
                    if D_t < D_c:
                        r_ = r_ *0
                    elif D_t >1.2*D_c:
                        r_ = r_*1.2
                    else:
                        r_ = r_*( 1-(D_c-D_t)/D_c )    # [r, 1.2r]

        else:
            raise RuntimeError() 
        return r_

    def statValue(self, stat, time, seq_len, mapping = False):
        if self.step_mapping: mapping = True
        time = time + self.baseTime / self.max_seq_len * seq_len if self.runtime_reward else 1
        initStats = self.initMapStats if mapping else self.initStats
        if   self.optimize == 'area':  return (1 - stat[0] / initStats[0]) / time
        elif self.optimize == 'delay': return (1 - stat[1] / initStats[1]) / time
        else: return (1 - stat[0] / initStats[0] * stat[1] / initStats[1]) / time     # mix


    def add_abc_alias(self, alias_flag):
        if alias_flag in [None, 'None']: return
        
        if (self.map_mode == 'SCL') and (alias_flag == 'alias_mapD_mst'):
            str_1, str_2 = 'alias mapD_mst  ', '"ifraig;map;mfs2;topo;strash"'
            assert self.map_arg != ''  
            str_2 = ('map -D %s'%(self.map_arg)).join(str_2.split("map"))
            self.abc.run(str_1+str_2)
            print(str_1+str_2)
        else:
            raise RuntimeError()

# class AbcEnvGraph(AbcEnv):
#     def __init__(self, bench, actions='-z;-l', optimize='mix', baseline='resyn2;resyn2', max_seq_len=20,
#                  seq_end='time', runtime_reward=False, mapping='', cell_lib='', state_ver=0, graph_ver='2.1'):
#         super().__init__(bench, actions, optimize, baseline, max_seq_len, seq_end, runtime_reward, mapping, cell_lib, state_ver)
#         '''
#         g_tver: node type encoding
#           0: combinational only
#           1: supports sequential
#           2: uses CI/CO instead of PI/PO/BI/BO
#         g_iver: inverter encoding
#           0: as node type
#           1: as node feature
#           2: as edge feature
#         '''
#         self.g_tver, self.g_iver = (int(a) for a in graph_ver.split('.'))
#         maxNodes = self.initStats[0] * 2
#         maxEdges = self.initStats[1] * 2
#         graphFeats = (6,9,7)[self.g_tver] - self.g_iver
#         spaces = {
#                 'state': self.observation_space,
#                 'gdim': gym.spaces.Box(0, maxEdges, shape=(2,)),
#                 'esrc': gym.spaces.Box(0, maxNodes, shape=(maxEdges,)),
#                 'edst': gym.spaces.Box(0, maxNodes, shape=(maxEdges,)),
#                 'ndfeat': gym.spaces.Box(-1, 1, shape=(maxNodes, graphFeats)),
#             }
#         if self.g_iver == 2:
#             spaces['ewgt'] = gym.spaces.Box(-1, 1, shape=(maxEdges,))

#         self.observation_space = gym.spaces.Dict(spaces)

#     def state(self):
#         combined = super().state()
#         esrc, edst, ewgt, ndfeat = self.abc.graphData(self.g_tver, self.g_iver)
#         edges, nodes = esrc.shape[0], ndfeat.shape[0]
#         maxEdges = self.observation_space['esrc'].shape[0]
#         maxNodes = self.observation_space['ndfeat'].shape[0]
#         assert edges <= maxEdges and nodes <= maxNodes and edges < 2 ** 24, f"graph too big. {edges, nodes}"
#         gdim = np.array([edges, nodes])
#         esrc.resize(self.observation_space['esrc'].shape)
#         edst.resize(self.observation_space['edst'].shape)
#         ndfeat.resize(self.observation_space['ndfeat'].shape)
#         states = {'state': combined, 'gdim': gdim, 'esrc': esrc, 'edst': edst, 'ndfeat': ndfeat}
#         if 'ewgt' in self.observation_space:
#             ewgt.resize(self.observation_space['ewgt'].shape)
#             states['ewgt'] = ewgt
#         return states


class AbcEnv_ASIC_step(AbcEnv_ASIC):
    pass

