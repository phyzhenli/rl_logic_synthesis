
import numpy as np
import gym
from os.path import abspath, expanduser
import subprocess
import re
import time
import shutil
import yaml

class AbcExeOpt(gym.Env):
    """ EDA environment """
    def __init__(self, options_yaml_file, runtime_reward=False, state_ver=0):

        with open(options_yaml_file, 'r') as options_f:
            options = yaml.safe_load(options_f)

        self.abc_exe = abspath(expanduser(options['abc_exe']))
        self.init_bench = abspath(expanduser(options['init_bench']))
        self.step_bench = abspath(expanduser(options['step_bench']))
        self.actions = options['actions']
        print(self.actions)
        self.optimize = options['optimize']
        self.baseline = options['baseline']
        print('baseline: ', self.baseline)
        self.max_seq_len = options['max_seq_len']
        self.seq_end = options['seq_end']
        self.state_ver = state_ver
        self.runtime_reward = runtime_reward
        self.runTime = 1e-20

        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

        self.initStats = self.getInitStats()
        self.getBaseTime()
        baseStats = self.getBaseStats()
        totalReward = self.statValue(baseStats, self.baseTime, self.max_seq_len)
        initStats = self.initStats
        # self.rewardBaseline = totalReward / self.max_seq_len
        print(f"\ninit {initStats} baseline {baseStats} base time {self.baseTime}  total reward {totalReward}")

    def getBaseTime(self):
        tt = 0; n = 3
        self.maxBaseTime = 0
        cmds = 'read ' + self.init_bench + '; strash; ' + self.baseline
        for i in range(n):
            tic = time.perf_counter()
            subprocess.run([self.abc_exe, '-c', cmds], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
            toc = time.perf_counter()
            t = toc-tic
            self.maxBaseTime = max(self.maxBaseTime, t)
            tt += t
        self.baseTime = tt / n

    def reset(self, *, seed = None, return_info = False, options = None):
        #super().reset(seed=seed)
        self.runTime = 1e-20
        self.lastStats = self.getInitStats() # The initial AIG statistics
        self.curStats = self.lastStats  # the current AIG statistics
        shutil.copy(self.init_bench, self.step_bench)
        self.actTime = [(0,0)] * self.numActions()  # average action runtimes
        self.actSeq = []                # action sequence
        if return_info:
            return self.state(), {}
        else:
            return self.state()

    def numActions(self):
        return len(self.actions)

    def lastAction(self, i):
        return self.actSeq[-i] if i < len(self.actSeq) else -1

    def getInitStats(self):
        cmds = 'read ' + self.init_bench + '; strash; print_stats; '
        subp_log = subprocess.run([self.abc_exe, '-c', cmds], stdout=subprocess.PIPE, text=True)
        abc_log = subp_log.stdout.replace('\x1b[1;37m', '').replace('\x1b[0m', '')
        for line in abc_log.split('\n'):
            if 'i/o' in line:
                nd = int( (re.search('and *= *(\d+) *', line))[1] )
                lev = int( (re.search('lev *= *(\d+) *', line))[1] )
        return nd, lev
    
    def getBaseStats(self):
        cmds = 'read ' + self.init_bench + '; strash; '
        cmds += self.baseline + "; "
        cmds += 'print_stats; '
        subp_log = subprocess.run([self.abc_exe, '-c', cmds], stdout=subprocess.PIPE, text=True)
        abc_log = subp_log.stdout.replace('\x1b[1;37m', '').replace('\x1b[0m', '')
        for line in abc_log.split('\n'):
            if 'i/o' in line:
                nd = int( (re.search('and *= *(\d+) *', line))[1] )
                lev = int( (re.search('lev *= *(\d+) *', line))[1] )
        return nd, lev
    
    def getStepStats(self):
        cmds = 'read ' + self.step_bench + '; strash; print_stats; '
        subp_log = subprocess.run([self.abc_exe, '-c', cmds], stdout=subprocess.PIPE, text=True)
        abc_log = subp_log.stdout.replace('\x1b[1;37m', '').replace('\x1b[0m', '')
        for line in abc_log.split('\n'):
            if 'i/o' in line:
                nd = int( (re.search('and *= *(\d+) *', line))[1] )
                lev = int( (re.search('lev *= *(\d+) *', line))[1] )
        return nd, lev

    def runAction(self, action):
        cmds = 'read ' + self.step_bench + '; strash; '
        cmds += action + "; "
        cmds += 'write_verilog ' + self.step_bench + '; '
        tic = time.perf_counter()
        subprocess.run([self.abc_exe, '-c', cmds], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
        toc = time.perf_counter()
        return toc - tic

    def step(self, actionIdx):
        self.actSeq.append(actionIdx)
        self.lastTime = self.runTime
        an, at = self.actTime[actionIdx]
        t = self.runAction(self.actions[actionIdx])
        self.actTime[actionIdx] = an+1, (an*at + t)/(an+1)
        self.runTime += self.actTime[actionIdx][1]
        # done = self.runTime > self.maxBaseTime if self.seq_end == 'time' else len(self.actSeq) >= self.max_seq_len
        done = len(self.actSeq) >= self.max_seq_len or self.seq_end == 'time' and self.runTime > self.maxBaseTime
        # update the statitics
        self.lastStats = self.curStats
        self.curStats = self.getStepStats()
        nextState = self.state()
        reward = self.reward(done)
        cmdSeq = '; '.join([self.actions[id] for id in self.actSeq])
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
        # base = 0 if self.seq_end == 'time' else self.rewardBaseline

        # print(curVal - lastVal)
        return curVal - lastVal

    def statValue(self, stat, time, seq_len, mapping = False):
        time = time + self.baseTime / self.max_seq_len * seq_len if self.runtime_reward else 1
        initStats = self.initStats
        if   self.optimize == 'area':  return (1 - stat[0] / initStats[0]) / time
        elif self.optimize == 'delay': return (1 - stat[1] / initStats[1]) / time
        else: return (1 - stat[0] / initStats[0] * stat[1] / initStats[1]) / time     # mix
