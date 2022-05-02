#!/usr/bin/env python3

# Copyright (c) 2019 MindAffect B.V.
#  Author: Jason Farquhar <jadref@gmail.com>
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
from argparse import Namespace
import os
import json
import re
from mindaffectBCI.decoder.utils import search_directories_for_file

def strip_comments(json_like):
    """
    Removes C-style comments from *json_like* and returns the result.
    """
    comments_re = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    def replacer(match):
        s = match.group(0)
        if s[0] == '/': return ""
        return s
    return comments_re.sub(replacer, json_like) 

def load_config(config_file):
    """load an online-bci configuration from a json file

    Args:
        config_file ([str, file-like]): the file to load the configuration from. 
    """    
    if isinstance(config_file,str):
        # search for the file in py-dir if not in CWD
        if not os.path.splitext(config_file)[1] == '.json':
            config_file = config_file + '.json'
        pydir = os.path.dirname(os.path.abspath(__file__))
        config_file = search_directories_for_file(config_file,pydir,os.path.join(pydir,'config'))
        print("Loading config from: {}".format(config_file))
        with open(config_file,'r',encoding='utf8') as f:
            config = f.read()            
    else:
        print("Loading config from: {}".format(config_file))
        config = config_file.read()
    # strip the comments
    config = strip_comments(config)
    # load the cleaned file
    config = json.loads(config)

    # set the label from the config file
    if 'label' not in config or config['label'] is None:
        # get filename without path or ext
        config['label'] = os.path.splitext(os.path.basename(config_file))[0]
        
    return config

def askloadconfigfile():
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        root = Tk()
        root.withdraw()
        pydir = os.path.dirname(os.path.abspath(__file__))
        initialdir = os.path.join(pydir,'config') if os.path.isdir(os.path.join(pydir,'config')) else pydir
        filename = askopenfilename(initialdir=initialdir,
                                    title='Chose mindaffectBCI Config File',
                                    filetypes=(('JSON','*.json'),('All','*.*')))
    except:
        print("Can't make file-chooser dialog, and no config file specified!  Aborting")
        raise ValueError("No config file specified")
    return filename


def set_args_from_dict(args:Namespace, config:dict):
    # MERGE the config-file parameters with the command-line overrides
    for name,val in config.items(): # config file parameter
        if name in args: # command line override available
            newval = getattr(args, name)
            if newval is None: # ignore not set
                pass
            elif isinstance(val,dict): # dict, merge with config-file version
                val.update(newval)
            else: # otherwise just use the command-line value
                val = newval
        setattr(args,name,val)
    return args

