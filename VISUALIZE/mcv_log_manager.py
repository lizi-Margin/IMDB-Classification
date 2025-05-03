import numpy as np
from UTIL.colorful import *

def get_a_logger():
    from VISUALIZE.mcom import mcom, logdir
    mcv = mcom( path='%s/logger/'%logdir,
                    digit=16,
                    rapid_flush=True,
                    draw_mode='Img',
                    tag='[lstm.py]',
                    resume_mod=False)
    mcv.rec_init(color='k')
    return mcv


class LogManager():
    def __init__(self, mcv=None, who=''):
        self.who = who
        if mcv is None:
            self.mcv = get_a_logger()
        else:
            self.mcv = mcv
        self.trivial_dict = {}
        self.smooth_trivial_dict = {}

    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict: self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])
        
        print_buf = [f'[{self.who}] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, '%(key, self.trivial_dict[key]))
            if self.mcv is not None:  
                alpha = 0.98
                if key in self.smooth_trivial_dict:
                    self.smooth_trivial_dict[key] = alpha*self.smooth_trivial_dict[key] + (1-alpha)*self.trivial_dict[key]
                else:
                    self.smooth_trivial_dict[key] = self.trivial_dict[key]
                self.mcv.rec(self.trivial_dict[key], key)
                self.mcv.rec(self.smooth_trivial_dict[key], key + ' - smooth')
        if print: print紫(''.join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()

        self.trivial_dict = {}