"""
This module is used to download word frequency dictionaries used for word correction.
It contains the following functions:

"""

#  Copyright (c) 2021,
#  Authors: Thomas de Lange, Thomas Jurriaans, Damy Hillen, Joost Vossers, Jort Gutter, Florian Handke, Stijn Boosman
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from mindaffectBCI.examples.presentation.smart_keyboard.app_exceptions import Logger
import requests as req
import sys
import os


def check_dictionary(language):
    """
    Function to make sure the dictionary for the specified language is downloaded.

    """

    # If the dictionary/frequency_lists folder does not exist yet, make it.
    basedirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(basedirectory,"dictionaries")
    if not os.path.isdir(path):
        print(os.path.join("Directory \'.", path) + " does not exist yet. Creating it now.")
        os.mkdir(path)

    path = os.path.join(path, "frequency_lists")
    if not os.path.isdir(path):
        print(os.path.join("Directory \'.", path) + " does not exist yet. Creating it now.")
        os.mkdir(path)

    # If the dictionary for the language has not been downloaded yet, do it.
    if not os.path.isfile(os.path.join(basedirectory,"dictionaries", "frequency_lists", language + ".txt")):
        download(language=language, full=False)


def download(language, full):
    """
    Function to download dictionaries.
    """

    # construct url and read it using request library
    init = 'https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/'

    if full:
        url = init + language.lower() + '/' + language.lower() + '_full.txt'
    else:
        url = init + language.lower() + '/' + language.lower() + '_50k.txt'

    try:
        # request the file
        resp = req.get(url)

        # check if request is successful
        if resp.status_code != 200:
            print('Could not find a frequency list for ' + language, file=sys.stderr)
        else:
            basedirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            direct = os.path.join(basedirectory, "dictionaries", "frequency_lists", language + '.txt')
            with open(direct, 'wb') as file:
                file.write(resp.content)
    except:
        Logger.log_download_frequency_list_error()
