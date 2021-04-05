import numpy as np


class Iupdater:
    def __init__(self):
        pass

class ColMaker:
    def __init__(self):
        self.param = {}

    def set_param(self, name, value):
        self.param[name] = value

    def update(self, gpair, p_fmat, trees):
        self.init_data()

    def init_data(self, gpair, p_fmat, root_index, tree):
        assert self.param['num_nodes'] == self.param['num_roots'], \
            "ColMaker: can only grow new tree"

class Entry:
    def __init__(self, index, fvalue):
        self.index = index
        self.fvalue = fvalue

    def cmp_value(self, other):
        return self.fvalue < other.fvalue


class Inst:
    def __init__(self, entries, length):
        self.data = entries
        self.length = length

    def __getitem__(self, item):
        return self.data[item]

class SparseBatch:
    def __init__(self):




Entry = {'index':}
SparseBatch = {}