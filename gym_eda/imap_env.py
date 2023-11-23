import imap
from wurlitzer import pipes
import re
import gym
import numpy as np

class EngineIMAP():
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        
        self.srewrite = "rewrite"
        self.sbalance = "balance"
        self.srefactor = "refactor"
        self.slut_opt = "lut_opt"
        self.smap_fpga = "map_fpga"
        self.shistory = "history"

        # store the optimization sequence
        self.sequence = []
    
    def read(self):
        imap.read_aiger(filename = self.input_file)

    def write(self):
        with open(self.output_file, 'w') as file:
            for command in self.sequence:
                file.write(command + '; ')
    
    def add_sequence(self, str):
        self.sequence.append(str)
    
    def rewrite(self, priority_size = 10, cut_size = 4, level_preserve = False, zero_gain = False, verbose = False):
        imap.rewrite(priority_size = priority_size, cut_size = cut_size, level_preserve = level_preserve, zero_gain = zero_gain, verbose = verbose)
        
    def refactor(self, max_input_size = 10, max_cone_size = 16, level_preserve = False, zero_gain = False, verbose = False):
        imap.refactor(max_input_size = max_input_size, max_cone_size = max_cone_size, level_preserve = level_preserve, zero_gain = zero_gain, verbose = verbose)
        
    def balance(self, verbose = False):
        imap.balance(verbose = verbose)
        
    def lut_opt(self, priority_size = 10, cut_size = 6, global_area_iterations = 1, local_area_iterations = 2, zero_gain = False, verbose = False):
        imap.lut_opt(priority_size = priority_size, cut_size = cut_size, global_area_iterations = global_area_iterations, local_area_iterations = local_area_iterations, zero_gain = zero_gain, verbose = verbose)

    def history(self, clear = False, size = False, add = False, replace = -1, backup = -1):
        imap.history(clear = clear, size = size, add = add, replace = replace, backup = backup)

    def map_fpga(self, priority_size = 10, cut_size = 6, global_area_iterations = 1, local_area_iterations = 1, type = 0, verbose = False):
        imap.map_fpga(priority_size = priority_size, cut_size = cut_size, global_area_iterations = global_area_iterations, local_area_iterations = local_area_iterations, type = type, verbose = verbose)   

    def clean_up(self, verbose = False):
        imap.cleanup(verbose = verbose)

    def ps(self, type = 0):
        with pipes() as (out, err):
            imap.print_stats(type = type)
        res  = out.read()
        pis  = int( (re.search('pis=(\d+),',    res))[1] )
        pos  = int( (re.search('pos=(\d+),',    res))[1] )
        area = int( (re.search('area=(\d+),',   res))[1] )
        dep  = int( (re.search('depth=(\d+)\n', res))[1] )
        return {'pis': pis, 'pos': pos, 'area': area, 'depth': dep}

class iMAPEnv(gym.Env):
    def __init__(self, input_file, optimize='mix', mapping='FPGA; 6', step_map=False, max_seq_len=5, state_ver=0) -> None:
        self.input_file = input_file
        self.imap = EngineIMAP(input_file, input_file+'.seq')
        self.actions = ['balance', 'rewrite', 'rewrite -z', 'refactor']
        print(f'\n-- space: {self.actions}')

        self.optimize = optimize
        self.mapping = mapping
        self.step_map = step_map
        self.max_seq_len = max_seq_len
        self.state_ver = state_ver

        self.observation_space = gym.spaces.Box(0, 10, shape=(self.dimState(),))
        self.action_space = gym.spaces.Discrete(self.numActions())

        self.imap.read()
        self.imap.history(add=True)
        self.initAigStats = self.getStats('')
        self.initMapStats = self.getStats(self.mapping)
        print(f"-- init AIG {self.initAigStats}, mapping to {self.mapping}, init {self.initMapStats}\n")

        self.best_imp = 0
        self.best_seq_file = input_file+'.seq'

    def reset(self, *, seed = None, return_info = False, options = None):
        self.imap.history(backup=1) # bug in history
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
            return self.imap.ps()
        else:
            mode, arg = mapping.split(';')
            if mode == 'FPGA':
                self.imap.map_fpga(cut_size=arg)
                return self.imap.ps(1)
            if mode == 'ASIC':
                pass

    def step(self, actionIdx):
        self.actSeq.append(actionIdx)
        if self.actions[actionIdx] == 'balance':
            self.imap.balance()
        if self.actions[actionIdx] == 'rewrite':
            self.imap.rewrite()
        if self.actions[actionIdx] == 'refactor':
            self.imap.refactor()
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
        nextState = self.state()
        reward = self.reward(done)
        cmdSeq = '; '.join([self.actions[id] for id in self.actSeq])
        if done:
            doneImp = self.statValue(self.curStats, done)
            if doneImp > self.best_imp:
                self.best_imp = doneImp
                with open(self.best_seq_file, 'w') as file:
                    file.write(cmdSeq+'; map_fpga')
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
