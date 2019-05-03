import sys
import os
HOME=os.environ['HOME']
sys.path.insert(1,HOME+'/github/StreamingSVM')


class Print:

    @staticmethod
    def err(skk): print("\033[91m {}\033[00m".format(skk))

    @staticmethod
    def console(skk): print("\033[92m {}\033[00m".format(skk))

    @staticmethod
    def info(skk): print("\033[93m {}\033[00m".format(skk))

    @staticmethod
    def result1(skk): print("\033[94m {}\033[00m".format(skk))

    @staticmethod
    def result2(skk): print("\033[95m {}\033[00m".format(skk))

    @staticmethod
    def info1(skk): print("\033[96m {}\033[00m".format(skk))

    @staticmethod
    def log(skk): print("\033[97m {}\033[00m".format(skk))

    @staticmethod
    def prBlack(skk): print("\033[98m {}\033[00m".format(skk))

    @staticmethod
    def overwrite(skk): print("\033[96m {}\033[00m".format(skk)) #,end='\r'
