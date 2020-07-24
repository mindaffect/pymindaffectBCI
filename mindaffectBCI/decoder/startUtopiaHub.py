import subprocess
import os

def run():
    pydir = os.path.dirname(os.path.abspath(__file__)) # mindaffectBCI/decoder/startUtopiaHub.py
    bindir = os.path.join(pydir,'..','..','bin') # ../../bin/

    # command to run the java hub
    cmd = ("java","-jar","UtopiaServer.jar")
    # args to pass to the java hub
    args = ("8400","0","../logs/mindaffectBCI.txt")

    # run the command, waiting until it has finished
    print("Running command: {}".format(cmd+args))
    utopiaHub = subprocess.run(cmd + args, cwd=bindir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Result: {}".format(utopiaHub.stdout))
    return utopiaHub

if __name__=="__main__":
    run()