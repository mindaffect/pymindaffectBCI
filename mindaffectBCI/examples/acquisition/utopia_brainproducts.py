import subprocess
import os
from time import sleep

def run(**kwargs):
    """startup the acquisation driver for a BrainProducts based amplifiers. 

    Note: This driver is based on the BrainProducts SDK -- and limited by the license 
    limitations of that SDK as described here <https://github.com/mindaffect/pymindaffectBCI/examples/acquisation/Licensing_Terms_BrainVision_Amplifer-SDK.pdf>.

    Returns:
        subprocess: subprocess object for communicating with the started amplifier driver.
    """    
    pydir = os.path.dirname(os.path.abspath(__file__)) # mindaffectBCI/examples/acquisation/utopia_brainproducts.py
    bindir = pydir

    # command to run the brainproducts driver
    cmd = ("bp2utopia",)
    # args to pass to the exe
    args = ()

    # run the command, waiting until it has finished
    print("Running command: {}".format(cmd+args))
    # TODO []: shell=True shouldn't be needed....
    acq = subprocess.Popen(cmd + args, shell=True, cwd=bindir)#, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sleep(1)
    print("Result: {}".format(acq.stdout))
    return acq

if __name__=="__main__":
    acq=run()
    acq.communicate()