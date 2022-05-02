#  Copyright (c) 2019 MindAffect B.V.
#  Author: Jason Farquhar <jadref@gmail.com>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

import subprocess
import os
from time import sleep
import signal

# set this to use a specific java exec location if available
javaexedir = '../../../../OpenJDKJRE64/bin'


def run(label='', logdir=None, port: int = 8400, verb: int = -1):
    pydir = os.path.dirname(
        os.path.abspath(__file__))  # mindaffectBCI/decoder/startUtopiaHub.py
    jardir = os.path.join(pydir, '..', 'hub')
    javaexe = os.path.join(
        pydir, javaexedir,
        'java') if javaexedir and os.path.exists(javaexedir) else 'java'

    # make the logs directory if not already there
    if logdir is None:
        logdir = os.path.join(pydir, '../../logs')
    logdir = os.path.expanduser(logdir)
    if not os.path.exists(logdir):
        try:
            os.makedirs(logdir)
        except:
            print("Error making the log directory {}".format(logdir))
    if not os.path.exists(logdir):
        logdir = pydir
    print("Saving to {}".format(logdir))

    # command to run the java hub
    cmd = (javaexe, "-jar", "UtopiaServer.jar")
    # args to pass to the java hub
    if label is not None:
        logfile = "mindaffectBCI_{}.txt".format(label)
    else:
        logfile = "mindaffectBCI.txt"
    args = ("{:d}".format(port), "{:d}".format(verb),
            os.path.join(logdir, logfile))

    # run the command, waiting until it has finished
    print("Running command: {} in dir {}".format(cmd + args, jardir))
    utopiaHub = subprocess.Popen(
        cmd + args, cwd=jardir, shell=False, stdin=subprocess.DEVNULL
    )  #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sleep(1)
    return utopiaHub


# setup signal handler to forward to child processes
signal.signal(signal.SIGINT, lambda signum, frame: shutdown())
signal.signal(signal.SIGTERM, lambda signum, frame: shutdown())


def shutdown():
    print('shutdown')
    hub.send_signal(signal.SIGTERM)
    exit(0)


if __name__ == "__main__":
    hub = run(logdir='~/Desktop/logs')
    while True:
        sleep(1)
