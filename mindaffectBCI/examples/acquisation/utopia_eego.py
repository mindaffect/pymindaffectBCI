import subprocess
import os

def run(**kwargs):
    pydir = os.path.dirname(os.path.abspath(__file__)) # mindaffectBCI/examples/acquisation/utopia_eego.py
    bindir = pydir #os.path.join(pydir,'..','..','..','bin') # ../../bin/

    # command to run the eego driver
    cmd = ("eego2utopia",)
    # args to pass to the exe
    args = () #"8400","0","../logs/mindaffectBCI.txt")

    # run the command, waiting until it has finished
    print("Running command: {}".format(cmd+args))
    # TODO []: shell=True shouldn't be needed....
    eego2utopia = subprocess.run(cmd + args, shell=True, cwd=bindir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Result: {}".format(eego2utopia.stdout))
    return eego2utopia

if __name__=="__main__":
    run()