#!/usr/bin/env python3
#  Copyright (c) 2019 MindAffect B.V. 
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
import pyglet
import os
import random
from mindaffectBCI.presentation.screens.basic_screens import Screen
from mindaffectBCI.noisetag import Noisetag
from mindaffectBCI.utopiaclient import PredictedTargetProb, PredictedTargetDist
from mindaffectBCI.decoder.utils import search_directories_for_file, intstr2intkey


def load_symbols(fn, replacements:dict={"<comma>":",", "<space>":"_"}):
    """load a screen layout from a text file

    Args:
        fn (str): file name to load from
        replacements (dict): dictionary of string replacements to apply

    Returns:
        symbols [list of lists of str]: list of list of the symbols strings
    """
    symbols = []

    # search in likely directories for the file, cwd, cwd/symbols pydir, projectroot 
    fn = search_directories_for_file(fn,os.path.dirname(os.path.abspath(__file__)),
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)),'symbols'),
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','symbols'),
                                        os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))

    with open(fn,'r', encoding='utf8') as f:
        for line in f:
            # skip comment lines
            if line.startswith('#'): continue
            # delim is ,
            line = line.split(',')
            # strip whitespace
            line = [ l.strip() for l in line if l is not None ]
            # None for empty strings
            line = [ l if not l == "" else None for l in line ]
            # strip quotes
            line = [ l.strip('\"') if l is not None else l for l in line ]
            # BODGE: replace the <comma> string with the , symbol'
            line = [ replacements.get(l,l) for l in line ]

            # add
            symbols.append(line)

    return symbols


#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-------------------------------------------------------------
class SelectionGridScreen(Screen):

    LOGLEVEL=0
    def __init__(self, window, noisetag=None, symbols=None, objIDs=1,
                 noisetag_mode:str = 'prediction', noisetag_args:dict={'nTrials':10, 'duration':5},
                 bgFraction:float=.2, instruct:str="", label:str=None,
                 clearScreen:bool=True, background_color:tuple=None, sendEvents:bool=True, liveFeedback:bool=True, 
                 optosensor:bool=True, inject_threshold: float = None, inject_noise:float=None,
                 fixation:bool=False, state2color:dict=None,
                 target_only:bool=False, show_correct:bool=True, show_newtarget_count:bool=None,
                 waitKey:bool=True, stimulus_callback=None, framerate_display:bool=True,
                 logo:str='MindAffect_Logo.png', media_directories:list=[], font_size:int=None, only_update_on_change:bool=True,
                 self_paced_intertrial:int=-1, self_paced_sentence:str='\n\n\n\nPause.  Press <space> to continue'):
        """Screen which shows a grid of symbols which will be flickered with a given stimulus-sequence
           and the approrpiate information sent-to and recieved from a running hub+decoder to implement an on-line or off-line BCI system.

        Args:
            window (pyglet.window): pyglet window to display the selection grid in
            noisetag (mindaffectBCI.Noisetag): Noisetag object which manages the stimulus-sequence order and communication to the on-line BCI components.
            noisetag_mode (str, optional): The mode to run the noisetag sequence in.  One of: 'calibration','prediction','single_trial'.  Defaults to 'prediction'.
            noisetag_args (dict, optional): Additional argument to pass to the noisetag when setting the sequence up.  See `Noisetag.startPrediciton` or `Noisetag.startCalibration` or the options available.  Examples include `nTrials`,`duration`,`waitduration`, etc.  Defaults to dict().
            symbols (list-of-list-of-str, optional): the grid of symbols to display, i.e. the foreground letters of the virtual-keys. Defaults to None.
            objIDs (list-of-int|int, optional): the set of BCI unique object identifiers to use for the symbols in the grid, OR the starting objectID to use. Defaults to 1.
            bgFraction (float, optional): fraction of empty space between the background of the 'virtual keys' of the on-screen grid. Defaults to .2.
            instruct (str, optional): instruction to show at the top of the screen. Defaults to "".
            label (str, optional): human readable label for this screen, used for example to automatically make menu entries, or for logging messages. Defaults to None.
            clearScreen (bool, optional): set to true to clear the screen to background color before every frame flip. Defaults to True.
            background_color (3-tuple, optional): the color to use for the background of the grid. Defaults to None
            sendEvents (bool, optional): set to true to send events about the stimulus state to the hub and BCI.  N.B. if false no events are sent and hence no BCI functionality will work. Defaults to True.
            liveFeedback (bool, optional): set to true to show live feedback, in the form of the key label color, of the BCIs current most likely target. Defaults to True.
            optosensor (bool, optional): set to true to show the optosensor box in the top left of the window. Defaults to True.
            fixation (bool, optional): set to true to show a red fixation cross in the center of the grid. Defaults to False.
            state2color (dict, optional): dictionary with the mapping from stimulus 'states' (as obtained from the stimulus-sequence obtained from the noisetag object) to the color of the grid virtual key  background.  Note: the following states have pre-defined meanings: 0-no-stimulus, 254-cued target, 255-feedback.  Defaults to None.
            target_only (bool, optional): set to true to only show the cued target and it's stimuli.  This is used for 'simple-calibration' where only the target is show in order to not distract the user with the flickering of the other keys. Defaults to False.
            show_correct (bool, optional): if a target is cued, then show with a * if the BCI correctly selects this cued target. Defaults to True.
            show_newtarget_count (bool, optional): set to true to show a count at the top of the screen of the number of trials completed this run. Defaults to None.
            waitKey (bool, optional): if true the wait for key press to finish the screen.  In this case this means a key-press automatically finishes the stimulus display. Defaults to True.
            stimulus_callback (_type_, optional): call-back function to call when the stimulus is updated. Defaults to None.
            framerate_display (bool, optional): set to true to show the framerate counter in the top-right of the display. Defaults to True.
            logo (str, optional): file name of an image to load and show as the logo the top right of the display. Defaults to 'MindAffect_Logo.png'.
            media_directories (list, optional): list of directories to search for media-resources, such as images or audio, when loading. Defaults to [].
            font_size (int, optional): size of the base font to use for the instructions etc.  Note: the grid cell keys font size is automatically computed to fit in the grid-cell. Defaults to None.
            only_update_on_change (bool, optional): set to true to only update the stimulus objects state when that state has changed.  This is an efficiency optimization to reduce the computational demands of the stimuli. Defaults to True.
            self_paced_intertrial (int, optional): time in seconds to *automatically* pause the stimuli between trials and wait for the user to press a key to continue. Defaults to -1.
            self_paced_sentence (str, optional): sentence to show on the screen when auto-pausing to wait for the user to continue. Defaults to '\n\n\n\nPause.  Press <space> to continue'.
        """
        self.window, self.symbols, self.noisetag, self.objIDs = ( window, symbols, noisetag, objIDs)
        self.sendEvents, self.liveFeedback, self.optosensor, self.framerate_display, self.logo, self.background_color, self.font_size, self.waitKey, self.stimulus_callback, self.target_only, self.show_correct, self.fixation, self.only_update_on_change, self.self_paced_intertrial, self.self_paced_sentence, self.media_directories, self.inject_threshold, self.inject_noise, self.instruct, self.bgFraction =\
            (sendEvents, liveFeedback, optosensor, framerate_display, logo, background_color, font_size, waitKey, stimulus_callback, target_only, show_correct, fixation, only_update_on_change, self_paced_intertrial, self_paced_sentence, media_directories, inject_threshold, inject_noise, instruct, bgFraction)
        self.noisetag_mode, self.noisetag_args = noisetag_mode, noisetag_args
        self.label = label if label is not None else self.__class__.__name__
        # ensure media directories has right format
        if self.media_directories is None: self.media_directories = []
        elif not hasattr(self.media_directories,'__iter__'): self.media_directories=[self.media_directories]
        # create set of sprites and add to render batch
        self.clearScreen=clearScreen
        self.isRunning=False
        self.isDone=False
        self.pause=False
        self.nframe=0
        self.last_target_idx=-1
        self.framestart = self.getTimeStamp()
        self.frameend = self.getTimeStamp()
        #self.stimulus_state = None
        self.prev_stimulus_state = None # used to track state changes for display optimization / trigger injection
        self.target_idx = None
        # N.B. noisetag does the whole stimulus sequence
        self.set_noisetag(noisetag)
        # register event handlers
        self.noisetag.addSelectionHandler(self.doSelection)
        self.noisetag.addNewTargetHandler(self.doNewTarget)
        self.noisetag.addPredictionHandler(self.doPrediction)
        self.noisetag.addPredictionDistributionHandler(self.doPredictionDistribution)
        if self.symbols is not None:
            self.set_grid(self.symbols, objIDs, bgFraction, sentence=instruct, logo=logo)
        self.liveSelections = None
        self.feedbackThreshold = .4
        self.last_target_idx = -1
        self.show_newtarget_count=show_newtarget_count

        # add new state to color mappings
        if state2color is not None:
            self.state2color.update(intstr2intkey(state2color))

    def reset(self):
        """reset the stimulus state
        """
        self.isRunning=False
        self.isDone=False
        self.nframe=0
        self.t0=-1
        self.pause=False
        self.self_paced_time_ms=0
        self.last_target_idx=-1
        #self.stimulus_state = None
        if self.show_newtarget_count is not None and not self.show_newtarget_count==False:
            self.show_newtarget_count=0 
        self.prev_stimulus_state = None #self.stimulus_state
        self.set_grid()
        # reset the noisetag if not already running
        if self.noisetag and not self.noisetag.isRunning():
            self.reset_noisetag()

    def reset_noisetag(self):
        print('noisetag_reset')
        if self.noisetag_mode.lower() == 'calibration':
            self.noisetag.modeChange("calibration")
            self.noisetag.startCalibration(**self.noisetag_args)
        elif self.noisetag_mode.lower() == 'prediction':
            print('noisetag::prediction {}'.format(self.noisetag_args))
            self.noisetag.modeChange("prediction")
            self.noisetag.startPrediction(**self.noisetag_args)
        elif self.noisetag_mode.lower() == 'single_trial':
            self.noisetag.startSingleTrial(**self.noisetag_args)
        else:
            self.noisetag.startFlickerWithSelection(**self.noisetag_args)

    def elapsed_ms(self):
        """get the elasped time in milliseconds since this screen started running

        Returns:
            float: elapsed time in milliseconds
        """        
        return self.getTimeStamp()-self.t0 if self.t0 else -1

    def getTimeStamp(self):
        return self.noisetag.getTimeStamp()

    def set_noisetag(self, noisetag:Noisetag):
        """set the noisetag object to use to get/send the stimulus state info

        Args:
            noisetag (Noisetag): the noise tag object
        """        
        self.noisetag=noisetag

    def setliveFeedback(self, value:bool):
        """set flag if we show live prediction feedback (blue-objects) during stimulus

        Args:
            value (bool): live feedback value
        """        
        self.liveFeedback=value

    def setliveSelections(self, value):
        """set if we show live selection information

        Args:
            value (bool): flag to show live selections
        """       
        self.liveSelections = value

    def setshowNewTarget(self, value):
        """set if we show when a new target happens

        Args:
            value (bool): flag to show new target info
        """       
        self.show_newtarget_count=value

    def get_idx(self,idx):
        """get a linear index into the self.objects or self.labels arrays from a (i,j) index into self.symbols

        Args:
            idx ([type]): index into symbols

        Returns:
            int: linear index
        """        
        ii=0 # linear index
        for i in range(len(self.symbols)):
            for j in range(len(self.symbols[i])):
                if self.symbols[i][j] is None: continue
                if idx==(i,j) or idx==ii :
                    return ii
                ii = ii + 1
        return None

    def getLabel(self,idx):
        """get the label object at given index

        Args:
            idx ([type]): the index

        Returns:
            pyglet.label: the label object
        """        
        ii = self.get_idx(idx)
        return self.labels[ii] if ii is not None else None

    def setLabel(self,idx,val:str):
        """set the text value of the label at idx

        Args:
            idx ([type]): [description]
            val (str): the label string
        """        
        ii = self.get_idx(idx)
        # update the label object to the new value
        if ii is not None and self.labels[ii]:
            self.labels[ii].text=val

    def setObj(self,idx,val):
        """set the object at the given index

        Args:
            idx ([type]): [description]
            val ([type]): the new object at this index
        """
        ii = self.get_idx(idx)
        if ii is not None and self.objects[ii]:
            self.objects[ii]=val

    def doSelection(self, objID:int):
        """do processing triggered by selection of an object, e.g. add letter to sentence

        Args:
            objID (int): the objId of the selected letter
        """        
        if self.liveSelections == True:
            if objID in self.objIDs:
                print("doSelection: {}".format(objID))
                symbIdx = self.objIDs.index(objID)
                sel = self.getLabel(symbIdx)
                sel = sel.text if sel is not None else ''
                text = self.update_text(self.sentence.text, sel)
                if self.show_correct and self.last_target_idx>=0:
                    text += "*" if symbIdx==self.last_target_idx else "_"
                self.set_sentence( text )

    def doNewTarget(self):
        """do processing triggered by starting a new-target sequence, e.g. update the trial counter
        """        
        if not self.show_newtarget_count is None and self.show_newtarget_count:
            self.show_newtarget_count = self.show_newtarget_count+1
            bits = self.sentence.text.split("\n")
            text = "\n".join(bits[:-1]) if len(bits)>1 else bits[0]
            self.set_sentence( "{}\n{}".format(text,self.show_newtarget_count-1) )
        
        # insert a user based wait if wanted
        if self.self_paced_intertrial>0 and self.elapsed_ms() > self.self_paced_intertrial*1000 + self.self_paced_time_ms :
            self.set_sentence( self.sentence.text + self.self_paced_sentence )
            self.pause = True

    def doPredictionDistribution(self, ptd:PredictedTargetDist):
        """do processing triggered by recieving a predicted target distribution
        """        
        pass
    def doPrediction(self, pred:PredictedTargetProb):
        """do processing triggered by recieving a target prediction
        """        
        pass

    def update_text(self,text:str,sel:str):
        """add 'sel' to the current text with some intelligent replacements

        Args:
            text (str): current text
            sel (str): text to add.  Replacements: <bkspc> -> remove char, <space> = " " etc.

        Returns:
            str: updated text
        """        
        # process special codes
        if sel in ('<-','<bkspc>','<backspace>'):
            text = text[:-1]
        elif sel in ('spc','<space>','<spc>'):
            text = text + ' '
        elif sel == '<comma>':
            text = text + ','
        elif sel in ('home','quit'):
            pass
        elif sel == ':)':
            text = text + ""
        else:
            text = text + sel
        return text

    def set_sentence(self, text):
        '''set/update the text to show at the top of the screen '''
        if type(text) is list:
            text = "\n".join(text)
        self.sentence.begin_update()
        self.sentence.text=text
        self.sentence.end_update()
    
    def set_framerate(self, text):
        '''set/update the text to show in the frame rate box'''
        if type(text) is list:
            text = "\n".join(text)
        self.framerate.begin_update()
        self.framerate.text=text
        self.framerate.end_update()

    def set_grid(self, symbols=None, objIDs=None, bgFraction:float=None, sentence:str=None, logo=None):
        """set/update the grid of symbols to be selected from

        Args:
            symbols ([type], optional): list-of-list-of-strings for the objects to show. Should be a 2-d array, can use None, for cells with no symbol in. Defaults to None.
            objIDs ([type], optional): list-of-int the BCI object IDs for each of the displayed symbols. Defaults to None.
            bgFraction (float, optional): background fraction -- fraction of empty space between selectable objects. Defaults to .3.
            sentence (str, optional): starting sentence for the top of the screen. Defaults to "What you type goes here".
            logo (str, optional): logo to display at the top of the screen. Defaults to None.
        """
        winw, winh=self.window.get_size()
        # tell noisetag which objIDs we are using
        if symbols is None:
            symbols = self.symbols
        if sentence is None:
            sentence = self.instruct
        if bgFraction is None:
            bgFraction = self.bgFraction
        if objIDs is None:
            objIDs = self.objIDs

        if isinstance(symbols, str):
            symbols = load_symbols(symbols)
        if isinstance(symbols[0],str):
            symbols=[symbols]

        self.symbols=symbols
        # Number of non-None symbols
        self.nsymb  = sum([sum([(s is not None and not s == '') for s in x ]) for x in symbols])

        if objIDs is not None:
            if not hasattr(objIDs,'__iter__'):
                self.objIDs = list(range(objIDs, objIDs+self.nsymb+1))
            else:
                self.objIDs = list(objIDs)  
        else:
            self.objIDs = list(range(1,self.nsymb+1))
            objIDs = self.objIDs
        if logo is None:
            logo = self.logo
        # get size of the matrix
        self.gridheight  = len(symbols) # extra row for sentence
        self.gridwidth = max([len(s) for s in symbols])
        self.ngrid      = self.gridwidth * self.gridheight

        # update the objIDs set for the noisetag object
        self.noisetag_args['objIDs']=self.objIDs
        self.noisetag.setActiveObjIDs(self.objIDs)

        self.objects=[None]*self.nsymb
        self.labels=[None]*self.nsymb
        self.batch = pyglet.graphics.Batch()
        self.grid_background_group = pyglet.graphics.OrderedGroup(0)
        self.background_group = pyglet.graphics.OrderedGroup(1)
        self.foreground_group = pyglet.graphics.OrderedGroup(2)

        # init the symbols list -- using the bottom 90% of the screen
        self.init_background(0, 0, winw, winh*.9, self.background_color)
        self.init_symbols(symbols, 0, 0, winw, winh*.9, bgFraction )
        # add the other bits
        self.init_opto()
        # sentence in top 10% of screen
        self.init_sentence(sentence, winw*.15, winh, winw*.7, winh*.1 )
        self.init_framerate()
        self.init_logo(logo)
        self.init_fixation(0, 0, winw, winh*.9)


    def init_background(self, x, y, w, h, color=None):
        # make an color block for the background
        self.background = None
        if color:
            self.background = self.init_sprite('', x, y, w, h, batch=self.batch, group=self.grid_background_group)[0]
            self.background.color = color


    def init_symbols(self, symbols, x, y, w, h, bgFraction:float=.1, font_size:int=None):
        """setup the display for the given symbols set

        Args:
            symbols ([type]): the symbols to show, inside the given display box
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            bgFraction (float, optional): fraction of empty space between objects. Defaults to .1.
            font_size (int, optional): label font size. Defaults to None.
        """
        # now create the display objects
        sw = int(w/self.gridwidth) # cell-width
        bgoffsetx = int(sw*bgFraction) # offset within cell for the button
        sh = int(h/self.gridheight) # cell-height
        bgoffsety = int(sh*bgFraction) # offset within cell for the button
        idx=-1
        for i in range(len(symbols)): # rows
            sy = y + (self.gridheight-i-1)*sh # top-edge symbol
            for j in range(len(symbols[i])): # cols
                # skip unused positions
                if symbols[i][j] is None or symbols[i][j]=="": continue
                idx = idx+1
                symb = symbols[i][j]
                sx = x + j*sw # left-edge symbol
                self.objects[idx], self.labels[idx] = self.init_target(symb, 
                                                       sx+bgoffsetx, sy+bgoffsety, 
                                                       int(sw-bgoffsetx*2), int(sh-bgoffsety*2),
                                                       i,j,font_size=font_size)

    def init_target(self, symb:str, x, y, w, h, i=None,j=None, font_size:int=None):
        """ setup the display of a single target 'button' inside the given display box 

        Args:
            symb (str): the symbol to show at this target
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            i,j (int): row/col of this target in the grid
            font_size (int, optional): [description]. Defaults to None.

        Returns:
            (sprite, label): background sprite, and foreground text label
        """        
        sprite, symb = self.init_sprite(symb, x, y, w, h)
        label = self.init_label(symb, x, y, w, h, font_size)
        return sprite, label

    def init_label(self, symb, x, y, w, h, font_size=None):
        """ setup the display of a single target foreground label inside the given display box 

        Args:
            symb (str): the symbol to show at this target
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            font_size (int, optional): [description]. Defaults to None.

        Returns:
            (sprite, label): background sprite, and foreground text label
        """        
        # add the foreground label for this cell, and add to drawing batch
        symb = str(symb)
        if font_size is None: font_size = self.font_size
        if font_size is None or font_size == 'auto':
            font_size = int(min(w,h)*.75*72/96/max(1,len(symb)))
        label=pyglet.text.Label(symb, font_size=font_size, x=x+w/2, y=y+h/2,
                                color=(255, 255, 255, 255),
                                anchor_x='center', anchor_y='center',
                                batch=self.batch, group=self.foreground_group)
        return label

    def init_sprite(self, symb, x, y, w, h, scale_to_fit=True, batch=None, group=None):
        """setup the background image for this target 'button', inside the given display box

        Args:
            symb ([type]): [description]
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            scale_to_fit (bool, optional): do we scale the image to fix in the display box. Defaults to True.

        Returns:
            sprite: the background image sprite
        """
        try : # symb is image to use for this button
            img = search_directories_for_file(symb,os.path.dirname(__file__),
                                              os.path.join(os.path.dirname(__file__),'images'),
                                              os.path.join(os.path.dirname(__file__),'..','images'),
                                              *self.media_directories)
            img = pyglet.image.load(img)
            symb = '.' # symb is a fixation dot
        except :
            # create a 1x1 white image for this grid cell
            img = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255)).create_image(2, 2)
        img.anchor_x = img.width//2
        img.anchor_y = img.height//2
        # convert to a sprite (for fast re-draw) and store in objects list
        # and add to the drawing batch (as background)
        sprite=pyglet.sprite.Sprite(img, x=x+w/2, y=y+h/2,
                                    batch=batch if batch else self.batch, 
                                    group=group if group else self.background_group)
        sprite.w = w
        sprite.h = h
        # re-scale (on GPU) to the size of this grid cell
        if scale_to_fit:
            sprite.update(scale_x=int(sprite.w)/sprite.image.width, scale_y=int(sprite.h)/sprite.image.height)
        return sprite, symb


    def init_framerate(self):
        """create the display label for the framerate feedback box
        """        
        winw, winh=self.window.get_size()
        # add the framerate box
        self.framerate=pyglet.text.Label("", font_size=12, x=winw, y=winh,
                                        color=(255, 255, 255, 255),
                                        anchor_x='right', anchor_y='top',
                                        batch=self.batch, group=self.foreground_group)

    def init_sentence(self,sentence, x, y, w, h, font_size=32):
        """create the display lable for the current spelled sentence, inside the display box

        Args:
            sentence ([type]): the starting sentence to show
            x (float): left-side of the display box
            y (float): right-side of the display box
            w (float): width of the display box
            h (float): height of the display box
            font_size (int, optional): [description]. Defaults to 32.
        """        
        self.sentence=pyglet.text.Label(sentence, font_size=font_size, 
                                        x=x, y=y, width=w, height=h,
                                        color=(255, 255, 255, 255),
                                        anchor_x='left', anchor_y='top',
                                        multiline=True,
                                        batch=self.batch, group=self.foreground_group)

    def init_logo(self,logo):
        """init the sprite for the logo display at the top right of the window

        Args:
            logo ([type]): [description]
        """        
        # add a logo box
        if isinstance(logo,str): # filename to load
            logo = search_directories_for_file(logo,os.path.dirname(__file__),
                                                os.path.join(os.path.dirname(__file__),'..','..'))
            try :
                logo = pyglet.image.load(logo)
                logo.anchor_x, logo.anchor_y  = (logo.width,logo.height) # anchor top-right 
                self.logo = pyglet.sprite.Sprite(logo, self.window.width, self.window.height-16)
            except :
                self.logo = None
        if self.logo:
            self.logo.batch = self.batch
            self.logo.group = self.foreground_group
            self.logo.update(x=self.window.width,  y=self.window.height-16,
                            scale_x=self.window.width*.1/self.logo.image.width, 
                            scale_y=self.window.height*.1/self.logo.image.height)

    def init_opto(self):
        """initialize the sprite for the opto-sensor display
        """        
        winw, winh=self.window.get_size()
        self.opto_sprite, _ = self.init_sprite(1,0,winh*.9,winw*.1,winh*.1)
        self.opto_sprite.visible=False

    def init_fixation(self,x,y,w,h):
        """initialize the sprite/test for the fixation point in the middle of the display
        """
        self.fixation_obj = None
        # add a fixation point in the middle of the grid
        if isinstance(self.fixation,str):
            # load the given image
            self.fixation_obj = self.init_sprite(self.fixation,x,y,w,h)[0]

        elif self.fixation is not None and not self.fixation == 0:
            # make a cross character, with size given by self.fixation
            font_size = 40 if self.fixation == True else self.fixation
            self.fixation_obj = self.init_label("+",x=x,y=y,w=w,h=h,font_size=font_size)
            self.fixation_obj.color = (255,0,0,255)


    def injectSignal(self,stim_state:int):
        """given a stimulus or targetstate compute the amplitude of signal to inject into fake-data streams

        Args:
            stim_state (int|list-of-int): either the targetState if target is set, or the full stimulus state for all objects if not.

        Returns:
            int|float: the amplitude of the signal to inject
        """
        if hasattr(stim_state,'__iter__'):
            # summed stimulus power
            amp = int(sum(stim_state))
        else:
            # direct state to amplitude mapping
            amp = int(stim_state*255) if 0 <= stim_state and stim_state <= 1 else int(stim_state)
        if self.inject_threshold:
            amp = max(0, amp-self.inject_threshold)
        if self.inject_noise:
            # add gaussian noise
            amp = max(0,amp + random.gauss(0,self.inject_noise))
        return amp

    def is_done(self):
        """test if this screen is finished displaying

        Returns:
            bool: is done status
        """
        if self.isDone:
            self.noisetag.modeChange('idle')
        return self.isDone

    # mapping from bci-stimulus-states to display color
    state2color={0:(5, 5, 5),       # off=grey
                 1:(255, 255, 255), # on=white
                 254:(0,255,0),     # cue=green
                 255:(0,0,255),     # feedback=blue
                 None:(100,0,0)}    # red(ish)=task
    """ dictionary mapping for object stimulus 'states' (keys as int) to the color (3-tuples) (of the background) to use for that object.
    """
    def update_object_state(self, idx:int, state:int):
        """update the idx'th object to stimulus state state

            N.B. override this method to implement different state dependent stimulus changes, e.g. size changes, background images, rotations

            This version changes the sprite color based on the requested state.  By either;
               1) getting the color to use from the state2color class dictionary, i.e color=self.state2color[state]
               2) setting the background lumaniance color to state (for floating point states), i.e. color= color*state
        Args:
            idx (int): index of the object to update
            state (int or float): the new desired object state
        """        
        if self.objects[idx]:
            if isinstance(state,int): # integer state, map to color lookup table
                self.objects[idx].color = self.state2color.get(state,self.state2color[None])
            elif isinstance(state,float): # float state, map to intensity
                self.objects[idx].color = tuple(int(c*state) for c in self.state2color[1])                
        if self.labels[idx]:
            self.labels[idx].color=(255,255,255,255) # reset labels

    def get_stimtime(self):
        """get the timestamp of the start of the last stimulus event, i.e. the last window flip for visual stimuli 

        Returns:
            int: time-stamp for the last stimulus start
        """        
        winflip = self.window.lastfliptime
        if winflip > self.framestart or winflip < self.frameend:
            print("Error: frameend={} winflip={} framestart={}".format(self.frameend,winflip,self.framestart))
        return winflip


    def draw(self, t):
        """draw the letter-grid with given stimulus state for each object.
        Note: To maximise timing accuracy we send the info on the grid-stimulus state
        at the start of the *next* frame, as by this time we have the accurate timing information from the subsequent display flip. """
        if not self.isRunning:
            self.isRunning=True
            self.t0 = self.getTimeStamp()
        self.framestart=self.getTimeStamp()
        if self.pause:
            #start again if key is pressed
            if self.window.last_key_press:
                self.window.last_key_press = None
                self.pause=False
                # remove the cue text from the sentence
                sentence = self.sentence.text
                sentence = sentence[:-len(self.self_paced_sentence)]
                self.set_sentence( sentence )
                # record the time the user ended the cue
                self.self_paced_time_ms = self.elapsed_ms()
            else:
                self.do_draw()
                return

        stimtime = self.get_stimtime()
        self.nframe = self.nframe+1
        if self.sendEvents:
            self.noisetag.sendStimulusState(timestamp=stimtime,injectSignal=self.injectSignal)

        # get the current stimulus state to show
        try:
            self.noisetag.updateStimulusState()
            stimulus_state, target_idx, objIDs, sendEvents=self.noisetag.getStimulusState()
            if stimulus_state:
                stimulus_state = [ s for s in stimulus_state ] # BODGE: deep copy the stim-state to avoid changing noisetag
            target_state = stimulus_state[target_idx] if target_idx>=0 and len(stimulus_state)>0 else -1
            if target_idx >= 0 : self.last_target_idx = target_idx
        except StopIteration:
            self.isDone=True
            return

        if self.waitKey:
            if self.window.last_key_press:
                self.key_press = self.window.last_key_press
                #self.noisetag.reset()
                self.isDone = True
                self.window.last_key_press = None

        # turn all off if no stim-state
        if stimulus_state is None:
            stimulus_state = [0]*len(self.objects)

        # do the stimulus callback if wanted
        if self.stimulus_callback is not None:
            self.stimulus_callback(stimulus_state, target_state)

        # draw the white background onto the surface
        if self.clearScreen:
            self.window.clear()
        # update the state
        # TODO[]: iterate over objectIDs and match with those from the
        #         stimulus state!
        for idx in range(min(len(self.objects), len(stimulus_state))):
            # set background color based on the stimulus state (if set)
            try:
                if self.target_only and not target_idx == idx :
                    stimulus_state[idx]=0
                # Optimization: only update if the state has changed
                if not self.only_update_on_change: # default to always update
                    self.update_object_state(idx,stimulus_state[idx])
                else:
                    if not hasattr(self,'prev_stimulus_state') or self.prev_stimulus_state is None or not self.prev_stimulus_state[idx] == stimulus_state[idx]:
                        self.update_object_state(idx,stimulus_state[idx])
            except KeyError:
                pass


        # show live-feedback (if wanted)
        if self.liveFeedback:
            # get prediction info if any
            predMessage=self.noisetag.getLastPrediction()
            if predMessage and predMessage.Yest in objIDs and predMessage.Perr < self.feedbackThreshold:
                predidx=objIDs.index(predMessage.Yest) # convert from objID -> objects index
                prederr=predMessage.Perr
                # BODGE: manually mix in the feedback color as blue tint on the label
                if self.labels[predidx]:
                    fbcol=list(self.labels[predidx].color) 
                    fbcol[0]=int(fbcol[0]*.4) 
                    fbcol[1]=int(fbcol[1]*.4) 
                    fbcol[2]=int(fbcol[2]*.4+255*(1-prederr)*.6) 
                    self.labels[predidx].color = fbcol

        # disp opto-sensor if targetState is set
        if self.optosensor is not None and self.optosensor is not False and self.opto_sprite is not None:
            self.opto_sprite.visible=False  # default to opto-off
            
            opto_state = None
            if self.optosensor is True:
                # use the target state
                if target_state is not None and target_state >=0 and target_state <=1:
                    print("*" if target_state>.5 else '.', end='', flush=True)
                    opto_state = target_state
                else:
                    opto_state = sum(stimulus_state)

            elif isinstance(self.optosensor,int):
                # guarded array acess
                opto_state = stimulus_state[self.optosensor] if len(stimulus_state)>self.optosensor else None

            if opto_state is not None:
                self.opto_sprite.visible=True
                if opto_state <= 1.0: # pure on-off
                    self.opto_sprite.color = tuple(int(c*opto_state) for c in (255, 255, 255))
                else: # level -> intensity
                    self.opto_sprite.color = tuple(min(255,int(c*opto_state/100)) for c in (255,255,255))


        # record the previous and current stimulus state
        self.prev_stimulus_state = [s for s in stimulus_state] if stimulus_state else None # copy! old stimulus-state
        #self.stimulus_state = stimulus_state # current stimulus-state
        self.target_idx = target_idx

        # do the draw
        self.do_draw()


    def do_draw(self):
        """do the actual drawing of the current display state
        """
        if self.clearScreen:
            self.window.clear()

        self.batch.draw()
        #if self.logo: self.logo.draw()
        #if self.fixation_obj: self.fixation_obj.draw()
        self.frameend=self.getTimeStamp()
        # add the frame rate info
        # TODO[]: limit update rate..
        if self.framerate_display and hasattr(self.window,'flipstats'):
            self.window.flipstats.update_statistics()
            self.set_framerate("{:4.1f} +/-{:4.1f}ms".format(self.window.flipstats.median,self.window.flipstats.sigma))                




if __name__=='__main__':
    from mindaffectBCI.presentation.ScreenRunner import initPyglet, run_screen
    from mindaffectBCI.noisetag import Noisetag
    # make a noisetag object without the connection to the hub for testing
    nt = Noisetag(stimSeq='mgold_65_6532.txt', utopiaController=None)
    window = initPyglet(width=640, height=480)
    screen = SelectionGridScreen(window, nt, 
                        noisetag_mode='calibration', # set to run in calibration mode
                        symbols=[['1','2'],['3','4']], # use a simple 2x2 grid
                        inject_threshold=5, 
                        inject_noise=5, 
                        objIDs=10) # set to use objIDs 10,11,12,13
    # run the screen with the flicker
    run_screen(window, screen)
