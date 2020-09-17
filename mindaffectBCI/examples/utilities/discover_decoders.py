#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
#  Author: Jason Farquhar <jason@mindaffect.nl>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from mindaffectBCI.utopiaclient import UtopiaClient

# connect to the mindaffectDecoder
def run():
  U=UtopiaClient()
  try:
    U.autoconnect(timeout_ms=5000, queryifhostnotfound=False, scanifhostnotfound=True)
  except Exception as ex:
    print("Connection error: {}".format(ex))

  print("\n\n\n\n---------------------------------------------\n")
  if U.isConnected:
    print("Connected to decoder@ \n      *****               {}                     ****\n".format(U.gethostport()))
  else:
    print("Could not connect to decoder.. is it turned on? is it on the same wifi network?")
  print("\n---------------------------------------------\n\n\n")

if __name__=="__main__":
  run()
