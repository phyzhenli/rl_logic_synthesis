import numpy as np
import pdb
import gym

class LSEnv(gym.Env):
    """ Logic Synthesis environment """
    def __init__(self, bench, optimize, baseline, max_seq_len, seq_end,
                 runtime_reward, mapping, step_mapping, cell_lib, state_ver):
        self.bench = bench
        self.optimize = optimize
        self.baseline = baseline
        self.max_seq_len = max_seq_len
        self.seq_end = seq_end
        self.state_ver = state_ver
        self.runtime_reward = runtime_reward
        self.runTime = 1e-20
        if cell_lib: self.readCellLib(cell_lib)
        self.map_mode = ''
        if mapping:  self.map_mode, self.map_arg = mapping.split(';')
        self.step_mapping = step_mapping    # get mapping stats at each step instead of at each epoch

        self.initActions()
        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

        self.readBench()
        self.initStats = self.getStats()
        self.initMapStats = self.getStats(True)
        self.getBaseTime()
        baseStats = self.getStats(self.hasMapping())
        totalReward = self.statValue(baseStats, self.baseTime, max_seq_len, self.hasMapping())
        initStats = self.initMapStats if self.hasMapping() else self.initStats
        print(f"\ninit {initStats} baseline {baseStats} base time {self.baseTime}  total reward {totalReward}")

    def reset(self, *, seed = None, return_info = False, options = None):
        #super().reset(seed=seed)
        self.runTime = 1e-20
        self.readBench()
        self.lastStats = self.getStats()
        self.curStats = self.lastStats
        self.actTime = [(0,0)] * self.numActions()  # average action runtimes
        self.actSeq = []
        self.step_quality = []  # for step_mapping 
        if return_info:
            return self.state(), {}
        else:
            return self.state()

    def readCellLib(self, cell_lib): pass

    def getBaseTime(self):
        tt, n = 0, 3
        self.maxBaseTime = 0
        for i in range(n):
            self.readBench()
            t = self.runBaseline()
            self.maxBaseTime = max(self.maxBaseTime, t)
            tt += t
        self.baseTime = tt / n

    def hasMapping(self):
        return self.map_mode != ''

    def numActions(self):
        return len(self.actions)

    def lastAction(self, i):
        return self.actSeq[-i] if i < len(self.actSeq) else -1

    def actionText(self, i):
        return self.actions[i]

    def step(self, actionIdx):
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
        cmdSeq = '; '.join([self.actionText(id) for id in self.actSeq])
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
        if self.step_mapping: mapping = True
        time = time + self.baseTime / self.max_seq_len * seq_len if self.runtime_reward else 1
        initStats = self.initMapStats if mapping else self.initStats
        if   self.optimize == 'area':  return (1 - stat[0] / initStats[0]) / time
        elif self.optimize == 'delay': return (1 - stat[1] / initStats[1]) / time
        else: return (1 - stat[0] / initStats[0] * stat[1] / initStats[1]) / time     # mix
