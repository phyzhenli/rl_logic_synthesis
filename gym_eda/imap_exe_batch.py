import re
import gym
import numpy as np
from os.path import abspath, expanduser
import shutil
import subprocess
import yaml
import os

class iMAPExeBatch(gym.Env): # n_envs
    def __init__(self, imap_exe, benchs_yaml_file, step_file='rl_step.aig', optimize='mix', mapping='FPGA; 6', step_map=False, max_seq_len=10, state_ver=0) -> None:
        self.imap_exe = abspath(expanduser(imap_exe))
        with open(benchs_yaml_file, 'r') as f:
            self.benchs = [abspath(expanduser(b)) for b in yaml.safe_load(f)['benchs'] ]
        self.bench_index = 0
        self.step_file = abspath(expanduser(step_file))

        self.actions = ['balance',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 10', 'rewrite -z -P 12',
                        'refactor -I 10 -C 16', 'refactor -z -I 10 -C 16', 'refactor -I 10 -C 20', 'refactor -z -I 10 -C 20', 'refactor -I 12 -C 16', 'refactor -z -I 12 -C 16', 'refactor -I 12 -C 20', 'refactor -z -I 12 -C 20'
                    ]
        print(f'\n-- space: {self.actions}')
        print(f'\n-- benchmarks: {len(self.benchs)} {self.benchs}')

        self.mapping = mapping
        self.benchsAigStats = self.benchsMapStats = {}
        for b in self.benchs:
            shutil.copy(b, self.step_file)
            self.benchsAigStats[b] = self.getStats('')
            self.benchsMapStats[b] = self.getStats(self.mapping)

        self.optimize = optimize
        self.step_map = step_map
        self.max_seq_len = max_seq_len
        self.state_ver = state_ver

        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

    def reset(self, *, seed = None, return_info = False, options = None):
        bench = self.benchs[self.bench_index]
        shutil.copy(bench, self.step_file)
        self.initAigStats = self.benchsAigStats[bench]
        self.initMapStats = self.benchsMapStats[bench]
        if self.initAigStats['area'] < 5000: self.max_seq_len = 10
        elif 5000 <= self.initAigStats['area'] < 6000: self.max_seq_len = 9
        elif 6000 <= self.initAigStats['area'] < 7000: self.max_seq_len = 8
        elif 7000 <= self.initAigStats['area'] < 8000: self.max_seq_len = 7
        elif 8000 <= self.initAigStats['area'] < 9000: self.max_seq_len = 6
        else: self.max_seq_len = 5

        if self.step_map:
            self.lastStats = self.initMapStats
            self.curStats = self.initMapStats
        else:
            self.lastStats = self.initAigStats
            self.curStats = self.initAigStats
        self.actSeq = []                # action sequence
        if return_info:
            return self.state(), {}
        else:
            return self.state()

    def numActions(self):
        return len(self.actions)

    def lastAction(self, i):
        return self.actSeq[-i] if i < len(self.actSeq) else -1

    def getStats(self, mapping):
        if mapping == '':
            cmds = 'read_aiger -f ' + self.step_file + '; print_stats'
            subp_log = subprocess.run([self.imap_exe, '-c', cmds], stdout=subprocess.PIPE, text=True)
            imap_log = subp_log.stdout.strip()
            area = int( imap_log.split(', ')[-2].split('=')[-1] )
            depth = int( imap_log.split(', ')[-1].split('=')[-1] )
            return {'area': area, 'depth': depth}
        else:
            mode, arg = mapping.split(';')
            if mode == 'FPGA':
                cmds = 'read_aiger -f ' + self.step_file + '; map_fpga -C' + arg + '; print_stats -t 1'
                subp_log = subprocess.run([self.imap_exe, '-c', cmds], stdout=subprocess.PIPE, text=True)
                imap_log = subp_log.stdout.strip()
                area = int( imap_log.split(', ')[-2].split('=')[-1] )
                depth = int( imap_log.split(', ')[-1].split('=')[-1] )
                return {'area': area, 'depth': depth}
            if mode == 'ASIC':
                pass

    def step(self, actionIdx):
        self.actSeq.append(actionIdx)
        cmds = 'read_aiger -f ' + self.step_file + '; ' + self.actions[actionIdx] + "; "
        cmds += 'write_aiger -f ' + self.step_file + '; '
        # print(cmds)
        subprocess.run([self.imap_exe, '-c', cmds], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
        done = len(self.actSeq) >= self.max_seq_len
        # update the statitics
        self.lastStats = self.curStats
        if self.step_map:
            self.curStats = self.getStats(self.mapping)
        else:
            if done:
                self.curStats = self.getStats(self.mapping)
            else:
                self.curStats = self.getStats('')
        # print(self.curStats)
        nextState = self.state()
        reward = self.reward(done)
        cmdSeq = '; '.join([self.actions[id] for id in self.actSeq])
        if done:
            self.bench_index += 1
            self.bench_index = self.bench_index % len(self.benchs)
        return nextState, reward, done, {'area':self.curStats['area'], 'depth':self.curStats['depth'], 'seq':cmdSeq}

    def state(self):
        nHist = 3
        if self.step_map:
            stateArray = np.array([ self.curStats['area'] / self.initMapStats['area'],
                                    self.curStats['depth'] / self.initMapStats['depth'],
                                    self.lastStats['area'] / self.initMapStats['area'],
                                    self.lastStats['depth'] / self.initMapStats['depth'] ])
        else:
            stateArray = np.array([ self.curStats['area'] / self.initAigStats['area'],
                                    self.curStats['depth'] / self.initAigStats['depth'],
                                    self.lastStats['area'] / self.initAigStats['area'],
                                    self.lastStats['depth'] / self.initAigStats['depth'] ])
        lastOneHotActs  = np.zeros(self.numActions() + 1)
        for i in range(1, nHist + 1):
            lastOneHotActs[self.lastAction(i)] += 1/nHist
        # stepArray = np.array([self.runTime / self.baseTime, len(self.actSeq) / self.max_seq_len])
        if   self.state_ver == 0: result = np.concatenate((stateArray, lastOneHotActs), axis=-1)
        elif self.state_ver == 1: result = np.concatenate((stateArray, lastOneHotActs), axis=-1)
        elif self.state_ver == 4: result = np.concatenate((lastOneHotActs), axis=-1)
        elif self.state_ver == 2: result = np.concatenate((stateArray), axis=-1)
        elif self.state_ver == 3: result = stateArray
        return result.astype(np.float32)

    def dimState(self):
        lenState, lenOneHot, lenStep = 2, self.numActions() + 1, 2
        return (lenState+lenOneHot+lenStep, lenState+lenOneHot, lenState+lenStep, lenState, lenOneHot+lenStep)[self.state_ver]

    def reward(self, done):
        curVal  = self.statValue(self.curStats, done)
        lastVal = self.statValue(self.lastStats)
        return curVal - lastVal

    def statValue(self, stat, done = False):
        if self.step_map:
            initStats = self.initMapStats
        else:
            if done:
                initStats = self.initMapStats
            else:
                initStats = self.initAigStats
        if   self.optimize == 'area':  return (1 - stat['area'] / initStats['area'])
        elif self.optimize == 'delay': return (1 - stat['depth'] / initStats['depth'])
        else: return (1 - ( 0.6*stat['depth'] + 0.4*stat['area'] ) / ( 0.6*initStats['depth'] + 0.4*initStats['area'] ))