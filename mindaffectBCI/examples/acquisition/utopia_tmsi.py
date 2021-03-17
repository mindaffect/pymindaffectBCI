import subprocess
import os

def run(**kwargs):
    """startup the acquisation driver for a BrainProducts based amplifiers. 

    Note: This driver includes software developed by the eemagine Medical Imaging Solutions GmbH, 
    and is based on the AntNeuro SDK -- and limited by the license 
    limitations of that SDK as described here <https://github.com/mindaffect/pymindaffectBCI/examples/acquisation/UDO-SM-0124 rev7.1 eego amplifier software interface SDK user manual EN 2018-11-12.pdf>.
    In summary, this driver is provided without WARRANTY, for non-commercial use.

    Returns:
        subprocess: subprocess object for communicating with the started amplifier driver.
    """    
    pydir = os.path.dirname(os.path.abspath(__file__)) # mindaffectBCI/examples/acquisation/utopia_eego.py
    bindir = pydir #os.path.join(pydir,'..','..','..','bin') # ../../bin/

    # command to run the eego driver
    cmd = ("tmsi2utopia",)
    # args to pass to the exe
    args = () #"8400","0","../logs/mindaffectBCI.txt")

    # run the command, waiting until it has finished
    print("Running command: {}".format(cmd+args))
    # TODO []: shell=True shouldn't be needed....
    tmsi2utopia = subprocess.run(cmd + args, shell=True, cwd=bindir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Result: {}".format(tmsi2utopia.stdout))
    return tmsi2utopia

if __name__=="__main__":
    run()
