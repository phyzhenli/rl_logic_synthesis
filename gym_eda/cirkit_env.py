import cirkit_py as ckt
import numpy as np
import torch
import gym
import pdb


class CirkitEnv(gym.Env):
    __ID = 0
    #__log_file = open('CirkitEnv.log', 'w')
    actions = [
        lambda: ckt.mighty(area_aware=True),                                  # bz
        lambda: ckt.refactor(mig=True, strategy=1),                           # rf
        lambda: ckt.refactor(mig=True, strategy=1, zero_gain=True),           # rfz
        lambda: ckt.cut_rewrite(mig=True, lutsize=4, lutcount=25, multiple=True),                 # rw
        lambda: ckt.cut_rewrite(mig=True, lutsize=4, lutcount=25, multiple=True, zero_gain=True), # rwz
        lambda: ckt.resub(mig=True),                                          # rs
        #lambda: ckt.resub(mig=True, zero_gain=True),                          # rsz
        #lambda: ckt.resub(mig=True, max_pis=6),
        #lambda: ckt.resub(mig=True, max_pis=10),
        #lambda: ckt.resub(mig=True, max_pis=12),
    ]
    action_str = ['bz', 'rf', 'rfz', 'rw', 'rwz', 'rs']

    def __init__(self, bench, optimize='mix', max_seq_len=20,
                 seq_end='time', runtime_reward=False, lutsize=0, always_mapping=True, state_ver=0):
        #self.abc = AbcInterface()
        CirkitEnv.__ID += 1
        self.id = CirkitEnv.__ID
        self.log('__init__')
        self.bench = bench
        self.optimize = optimize
        self.max_seq_len = max_seq_len
        self.seq_end = seq_end
        self.state_ver = state_ver
        self.runtime_reward = runtime_reward
        self.lutsize = lutsize
        self.always_mapping = always_mapping
        self.runTime = 1e-20
        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

        self.readBench()
        self.initStats = self.getStats()
        self.initMapStats = self.getStats(True)
        self.getBaseTime()
        baseStats = self.getStats(self.hasMapping())
        totalReward = self.statValue(baseStats, self.baseTime, max_seq_len, self.hasMapping())
        initStats = self.initMapStats if self.hasMapping() else self.initStats
        print(f"\ninit {initStats} baseline {baseStats} base time {self.baseTime}  total reward {totalReward}\n")

    def log(self, s): pass
        #self.__log_file.write(f'{self.id}: {s}\n')
        #self.__log_file.flush()

    def readBench(self):
        if self.bench.endswith('.aig'):
            ckt.read_aiger(mig=True, filename=self.bench)
        elif self.bench.endswith('.blif'):
            ckt.read_blif(filename=self.bench)
            ckt.lut_resynthesis(mig=True)
        else:
            assert 0, "readBench: unknown file format"

    def numActions(self):
        return len(self.actions)

    def runAction(self, i):
        res = self.actions[i]()
        #print(i, res)
        return res["time_total"]

    def runSeq(self, seq):
        t = 0
        for i in seq:
            t += self.runAction(i)
        return t

    def runBaseline(self):
        self.readBench()
        # pdb.set_trace() # tmp_lst=[self.action_str[i_] for i_ in [0,3,1,0,3,4,0,2,4,0,0,3,1,0,3,4,0,2,4,0]]; print(tmp_lst)
        return self.runSeq([0,3,1,0,3,4,0,2,4,0,0,3,1,0,3,4,0,2,4,0])  # compress2; compress2
        # return self.runSeq([0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,])  # bz;rw repeated 10;

    def getBaseTime(self):
        tt, n = 0, 3
        self.maxBaseTime = 0
        for i in range(n):
            t = self.runBaseline()
            self.maxBaseTime = max(self.maxBaseTime, t)
            tt += t
        self.baseTime = tt / n

    def reset(self, *, seed = None, return_info = False, options = None):
        #super().reset(seed=seed)
        self.log('reset')
        self.runTime = 1e-20
        self.readBench()
        self.lastStats = self.getStats() # The initial AIG statistics
        self.curStats = self.lastStats  # the current AIG statistics
        self.actTime = [(0,0)] * self.numActions()  # average action runtimes
        self.actSeq = []                # action sequence
        # pdb.set_trace()   # todo
        if return_info:
            return self.state(), {}
        else:
            return self.state()

    def lastAction(self, i):
        return self.actSeq[-i] if i < len(self.actSeq) else -1

    def hasMapping(self):
        return self.lutsize > 0

    def getStats(self, run_mapping = False):
        # pdb.set_trace()
        if self.always_mapping: run_mapping = True
        if run_mapping and self.hasMapping():
            ckt.lut_mapping(mig=True, lutsize=self.lutsize)
            # ckt.satlut_mapping(mig=True, lutsize=6) # ?
            ckt.collapse_mapping(mig=True)
            res = ckt.ps(lut=True, silent=True)  # silent=True
        else:
            res = ckt.ps(mig=True, silent=True)  # silent=True
        return res["gates"], res["depth"]

    def step(self, actionIdx):
        self.log('step')
        self.actSeq.append(actionIdx)
        self.lastTime = self.runTime
        an, at = self.actTime[actionIdx]
        t = self.runAction(actionIdx)
        self.actTime[actionIdx] = an+1, (an*at + t)/(an+1)
        self.runTime += self.actTime[actionIdx][1]
        done = len(self.actSeq) >= self.max_seq_len or self.seq_end == 'time' and self.runTime > self.maxBaseTime
        # update the statitics
        self.lastStats = self.curStats
        self.curStats = self.getStats(done)
        nextState = self.state()
        reward = self.reward(done)
        cmdSeq = '; '.join([self.action_str[id] for id in self.actSeq])
        # pdb.set_trace()   # todo
        return nextState, reward, done, {'nd':self.curStats[0], 'lev':self.curStats[1], 'seq':cmdSeq}

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
        return curVal - lastVal

    def statValue(self, stat, time, seq_len, mapping = False):
        if self.always_mapping: mapping = True
        time = time + self.baseTime / self.max_seq_len * seq_len if self.runtime_reward else 1
        initStats = self.initMapStats if mapping else self.initStats
        if   self.optimize == 'area':  return (1 - stat[0] / initStats[0]) / time
        elif self.optimize == 'delay': return (1 - stat[1] / initStats[1]) / time
        else: return (1 - stat[0] / initStats[0] * stat[1] / initStats[1]) / time     # mix

