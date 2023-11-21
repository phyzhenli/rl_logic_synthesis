import cirkit_py as ckt
import numpy as np
import os
import pdb
import gym
from .ls_env import LSEnv
from abc_py import AbcInterface

class CirkitEnv(LSEnv):
    def __init__(self, bench, ntk_type='aig', optimize='mix', baseline='resyn2;resyn2', max_seq_len=20,
                 seq_end='time', runtime_reward=False, mapping='', cell_lib='', step_mapping = False, state_ver=0,
                 abc_lib='', abc_map_tail='abc_map_ver1', log_folder='',
                 ):
        self.ntk_type = ntk_type

        # for transfer to abc
        self.abc_lib = abc_lib
        self.abc_map_tail = abc_map_tail
        self.log_folder = log_folder
        self.step_quality = []
        self.abc = AbcInterface()
        self.abc.run('read ' +self.abc_lib)

        LSEnv.__init__(self, bench, optimize, baseline, max_seq_len, seq_end,
                       runtime_reward, mapping, step_mapping, cell_lib, state_ver)
        

    def initActions(self):
        base_actions  = ['bl', 'rf', 'rfz', 'rw', 'rwz', 'rs', 'rd']
        extra_actions = {'aig': [], 'mig': ['mighty', 'bzm'], 'xag': ['fanin_opt'], 'xmg': ['mighty', 'bzm']}
        self.actions = base_actions + extra_actions[self.ntk_type]

    def readBench(self):
        if self.bench.endswith('.aig'):
            file_type = 'aiger'
        elif self.bench.endswith('.blif'):
            file_type = 'blif'
        else:
            assert 0, "readBench: unknown file format"
        ckt.run(f'read_{file_type} --{self.ntk_type} {self.bench}')

    def readCellLib(self, cell_lib):
        ckt.run(f'read_genlib {cell_lib}')

    def runAction(self, i):
        res = ckt.run(self.actions[i])
        return res["time_total"]

    def runBaseline(self):
        res = ckt.run(self.baseline)
        return res["time_total"]

    def getStats(self, run_mapping = False):
        if self.step_mapping: run_mapping = True

        if run_mapping and self.map_mode == 'SCL':
            res = ckt.run(f'map -d {self.map_arg}; ps')
            ckt.run(f'ps --{self.ntk_type}')
            return res["area"], res["delay"]

        if run_mapping and self.map_mode == 'FPGA':
            res = ckt.run(f'lut_mapping -k {self.map_arg}; ps')
            ckt.run(f'ps --{self.ntk_type}')
            return res["gates"], res["depth"]

        elif run_mapping and self.map_mode == 'to_abc':
            # if not self.map_tail:
            res = ckt.run(f'lut_mapping -k {self.map_arg}; ps')
            # * ref test_map.py  
            blif_file = os.path.join(self.log_folder, 'ckt_write.blif')
            ckt.run('write_blif ' +blif_file)  # write_blif --klut 
            ckt.run(f'ps --{self.ntk_type}')  # to change ckt [store], to avoid error at next action, 

            self.abc.read(blif_file)
            # isaig, i, o, lat, nd, edge, lev = self.abc.ntkStats()
            print(self.abc.ntkStats())
            if self.abc_map_tail == 'abc_map_ver1':
                t = self.abc.run('dch; map')
            elif self.abc_map_tail == 'abc_map_ver2':
                t = self.abc.run('strash;ifraig;scorr;dc2;dretime;strash;dch -f;map;')
            else: raise RuntimeError()
            # print(f'{blif_file} abc_map', self.abc.sclStats()) # area, delay
            return self.abc.sclStats()
            
        else:
            res = ckt.run(f'ps --{self.ntk_type}')
            return res["gates"], res["depth"]


class CirkitEnv_step(CirkitEnv):
    
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
        self.step_quality.append((float(format(self.curStats[0], '.1f')), float(format(self.curStats[1], '.1f')))) # (area, delay) # write to file 
        return nextState, reward, done, {'nd':self.curStats[0], 'lev':self.curStats[1], 'seq':cmdSeq, 'step_q':str(self.step_quality).replace(',', '')}
