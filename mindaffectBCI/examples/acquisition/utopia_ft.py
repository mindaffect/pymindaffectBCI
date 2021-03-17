#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
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

def run(label='', logdir=None, **kwargs):
    pydir = os.path.dirname(os.path.abspath(__file__)) # mindaffectBCI/decoder/startUtopiaHub.py
    bindir = os.path.join(pydir) 

    # command to run the java hub
    jars = ("messagelib\\UtopiaServer.jar","ftbuffer\\lib\\BufferServer.jar","ftbuffer\\FT2Utopia.jar")
    classpath = ";".join([ os.path.join(bindir,jar) for jar in jars])
    mainclass = "nl.ma.utopia.FT2Utopia"
    cmd = ("java","-cp",classpath,mainclass)
    # args to pass to the utopia2ft
    # args-order: buffhostport utopiaport utopiatimeout buffertimeout ft2bufferONLY outdir
    args = ()

    # run the command, waiting until it has finished
    print("Running command: {}".format(cmd+args))
    acquisition = subprocess.run(cmd + args, cwd=bindir, shell=False)#,
                               #stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sleep(1)
    return acquisition

if __name__=="__main__":
    run()
