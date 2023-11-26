import re
import gym
import numpy as np
from os.path import abspath, expanduser
import shutil
import subprocess


class iMAPExe(gym.Env):
    def __init__(self, imap_exe, input_file, step_file='', optimize='mix', mapping='FPGA; 6', step_map=False, max_seq_len=20, state_ver=0) -> None:
        self.imap_exe = abspath(expanduser(imap_exe))
        self.input_file = abspath(expanduser(input_file))
        self.step_file = abspath(expanduser(step_file)) if step_file != '' else abspath(expanduser(input_file+'_step'))
        shutil.copy(self.input_file, self.step_file)
        self.actions = ['balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance',    'balance',
                        'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance',
                        'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance',
                        'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance',
                        'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance',
                        'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance', 'balance',

                        'rewrite -P 6', 'rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        'rewrite -P 6', 'rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        'rewrite -P 6', 'rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        'rewrite -P 6', 'rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        'rewrite -P 6', 'rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        'rewrite -P 6', 'rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        'refactor -I 6 -C 10', 'refactor -z -I 6 -C 10', 'refactor -I 6 -C 12', 'refactor -z -I 6 -C 12', 'refactor -I 6 -C 14', 'refactor -z -I 6 -C 14', 'refactor -I 6 -C 16', 'refactor -z -I 6 -C 16', 'refactor -I 6 -C 18', 'refactor -z -I 6 -C 18', 'refactor -I 6 -C 20', 'refactor -z -I 6 -C 20', 'refactor -I 8 -C 10', 'refactor -z -I 8 -C 10', 'refactor -I 8 -C 12', 'refactor -z -I 8 -C 12', 'refactor -I 8 -C 14', 'refactor -z -I 8 -C 14', 'refactor -I 8 -C 16', 'refactor -z -I 8 -C 16', 'refactor -I 8 -C 18', 'refactor -z -I 8 -C 18', 'refactor -I 8 -C 20', 'refactor -z -I 8 -C 20', 'refactor -I 10 -C 10', 'refactor -z -I 10 -C 10', 'refactor -I 10 -C 12', 'refactor -z -I 10 -C 12', 'refactor -I 10 -C 14', 'refactor -z -I 10 -C 14', 'refactor -I 10 -C 16', 'refactor -z -I 10 -C 16', 'refactor -I 10 -C 18', 'refactor -z -I 10 -C 18', 'refactor -I 10 -C 20', 'refactor -z -I 10 -C 20', 'refactor -I 12 -C 10', 'refactor -z -I 12 -C 10', 'refactor -I 12 -C 12', 'refactor -z -I 12 -C 12', 'refactor -I 12 -C 14', 'refactor -z -I 12 -C 14', 'refactor -I 12 -C 16', 'refactor -z -I 12 -C 16', 'refactor -I 12 -C 18', 'refactor -z -I 12 -C 18', 'refactor -I 12 -C 20', 'refactor -z -I 12 -C 20' # 48 refactor
                    ]
        self.actions = ['balance',
                        # 'rewrite -P 6','rewrite -P 8',
                        'rewrite -P 10', 'rewrite -P 12',
                        # 'rewrite -z -P 6', 'rewrite -z -P 8',
                        'rewrite -z -P 10', 'rewrite -z -P 12',

                        # 'refactor -I 6 -C 10', 'refactor -z -I 6 -C 10', 'refactor -I 6 -C 16', 'refactor -z -I 6 -C 16', 'refactor -I 6 -C 20', 'refactor -z -I 6 -C 20', 'refactor -I 10 -C 10', 'refactor -z -I 10 -C 10', 'refactor -I 10 -C 16', 'refactor -z -I 10 -C 16', 'refactor -I 10 -C 20', 'refactor -z -I 10 -C 20', 'refactor -I 12 -C 10', 'refactor -z -I 12 -C 10', 'refactor -I 12 -C 16', 'refactor -z -I 12 -C 16', 'refactor -I 12 -C 20', 'refactor -z -I 12 -C 20' # 18 refactor
                        'refactor -I 10 -C 16', 'refactor -z -I 10 -C 16', 'refactor -I 10 -C 20', 'refactor -z -I 10 -C 20', 'refactor -I 12 -C 16', 'refactor -z -I 12 -C 16', 'refactor -I 12 -C 20', 'refactor -z -I 12 -C 20' # 8 refactor
                    ]
        print(f'\n-- space: {self.actions}')

        self.optimize = optimize
        self.mapping = mapping
        self.step_map = step_map
        self.max_seq_len = max_seq_len
        self.state_ver = state_ver

        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

        self.initAigStats = self.getStats('')
        self.initMapStats = self.getStats(self.mapping)
        print(f"-- init AIG {self.initAigStats}, mapping to {self.mapping}, init {self.initMapStats}\n")

        # if self.initAigStats['area'] < 5000: self.max_seq_len = 10
        # if 5000 <= self.initAigStats['area'] < 6000: self.max_seq_len = 9
        # if 6000 <= self.initAigStats['area'] < 7000: self.max_seq_len = 8
        # if 7000 <= self.initAigStats['area'] < 8000: self.max_seq_len = 7
        # if 8000 <= self.initAigStats['area'] < 9000: self.max_seq_len = 6
        # if self.initAigStats['area'] >= 9000: self.max_seq_len = 5

        # self.best_imp = 0
        # self.best_seq_file = input_file+'.seq'
        # with open(self.best_seq_file, 'w') as file:
        #             file.write('balance; rewrite; refactor; map_fpga')


    def reset(self, *, seed = None, return_info = False, options = None):
        shutil.copy(self.input_file, self.step_file)
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
        # if done:
        #     doneImp = self.statValue(self.curStats, done)
        #     if doneImp > self.best_imp:
        #         self.best_imp = doneImp
        #         with open(self.best_seq_file, 'w') as file:
        #             file.write(cmdSeq+'; map_fpga')
        #     if self.best_imp == 0:
        #         with open(self.best_seq_file, 'w') as file:
        #             file.write('map_fpga')
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