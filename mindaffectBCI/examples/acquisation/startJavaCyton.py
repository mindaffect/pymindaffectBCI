import subprocess
import os

def run(label:str='', host:str='-', usb_port:str="com4", config=8, use_aux:int=0, serial_event:int=0, packet_size:int=-1, **kwargs):
    pydir = os.path.dirname(os.path.abspath(__file__)) # mindaffectBCI/examples/acquisation
    bindir = os.path.join(pydir,'..','..','..','bin') # ../../../bin/

    # command to run the java hub
    classpath = ";".join((os.path.join(bindir,"UtopiaServer.jar"),"jssc-2.9.2.jar","openBCI2utopia.jar"))
    cmd = ("java","-cp", classpath, "openBCI2utopia")
    # args to pass to the java hub
    args = (usb_port, host, str(config), str(use_aux), str(serial_event), str(packet_size))

    # run the command, waiting until it has finished
    print("Running command: {}".format(cmd+args))
    acquisation = subprocess.run(cmd + args, cwd=pydir, shell=False,
                               stdin=subprocess.DEVNULL)
    print("Result: {}".format(acquisation.stdout))
    return acquisation

if __name__=="__main__":
    run()