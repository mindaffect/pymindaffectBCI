mindaffectBCI
=============
This repository contains the python SDK code for the Brain Computer Interface (BCI) developed by the company `Mindaffect <https://mindaffect.nl>`_.

Quick Start
-----------

If you have just got your mindaffectBCI and are looking for general information on how to use it, checkout our `Wiki <https://github.com/mindaffect/General/wiki/First-time-use>`_.  


File Structure
--------------
This repository is organized roughly as follows:

 - `mindaffectBCI <mindaffectBCI>`_ - contains the python package containing the mindaffectBCI SDK.  Important modules within this package are:
 
   - `noisetag.py <mindaffectBCI/noisetag.py>`_ - This module contains the main API for developing User Interfaces with BCI control
   - `utopiaController.py <minaffectBCI/utopiaController.py>`_ - This module contains the application level APIs for interacting with the MindAffect Decoder.
   - `utopiaclient.py <mindaffectBCI/utopiaclient.py>`_ - This module contains the low-level networking functions for communicating with the MindAffect Decoder - which is normally a separate computer running the eeg analysis software.
   - stimseq.py -- This module contains the low-level functions for loading and codebooks - which define how the presented stimuli will look.

 - `examples <mindaffectBCI/examples/>`_ - contains python based examples for Presentation and Output parts of the BCI. Important sub-directories

   - `output <mindaffectBCI/examples/output/>`_ - Example output modules.  An output module translates BCI based selections into actions.
   - `presentation <mindaffectBCI/examples/presentation/>`_ - Example presentation modules.  A presentation module, presents the BCI stimulus to the user, and is normally the main UI.  In particular here we have:

     - `framerate_check.py <mindaffectBCI/examples/presentation/framerate_check.py>`_ - Which you can run to test if your display settings (particularly vsync) are correct for accurate flicker presentation.
     - `selectionMatrix.py <mindaffectBCI/examples/presentation/selectionMatrix.py>`_ - Which you can run as a simple example of using the mindaffectBCI to select letters from an on-screen grid.

   - `utilities <mindaffectBCI/examples/utilities/>`_ - Useful utilities, such as a simple *raw* signal viewer

Installing mindaffectBCI
------------------------

That's easy::

  pip3 install mindaffectBCI


Getting Support
---------------

For a general overview of how to use the mindaffectBCI, hardware, software and how to use it, see the `system wiki <https://github.com/mindaffect/General/wiki>`_.

If you run into and issue you can either directly raise an issue on the projects `github page <https://github.com/mindaffect/pymindaffectBCI>`_ or directly contact the developers on `gitter <https://gitter.im/mindaffect>`_ -- to complain, complement, or just chat:

.. image:: https://badges.gitter.im/mindaffect/unitymindaffectBCI.svg
   :target: https://gitter.im/mindaffect/pymindaffectBCI?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Testing the mindaffectBCI SDK
-----------------------------

This SDK provides the functionality needed to add Brain Controls to your own applications.  However, it *does not* provide the actual brain measuring hardware (i.e. EEG) or the brain-signal decoding algorithms. 

In order to allow you to develop and test your Brain Controlled applications without connecting to a real mindaffect Decoder, we provide a so called "fake recogniser".  This fake recogniser simulates the operation of the true mindaffect decoder to allow easy development and debugging.  Before starting with the example output and presentation modules.  You can download the fakerecogniser from our github page `bin directory <https://github.com/mindaffect/pymindaffectBCI/tree/master/bin>`_

You should start this fake recogniser by running, either ::

  bin/startFakeRecogniser.bat
  
if running on windows, or  ::

  bin/startFakeRecogniser.sh

if running on linux/macOS

If successfull, running these scripts should open a terminal window which shows the messages recieved/sent from your example application.

Note: The fakerecogniser is written in `java <https://www.java.com>`_, so you will need a JVM with version >8 for it to run.  If needed download from `here <https://www.java.com/ES/download/>`_

Quick Installation Test
-----------------------

You can run a quick test if the installation is correct by running::

  python3 -m mindaffectBCI.noisetag

Essentially, this run the SDK test code which pretends to run a full BCI sequence, with decoder discovery, calibration and prediction.  If you have the fakerecognise running then this should do this in a terminal and generate a lot of text saying things like: `cal 1/10`.

Quick BCI Test
--------------

If you have installed [pyglet](pyglet.org), e.g. using `pip3 install pyglet`, then you can also try some more advanced full BCI exmaples with stimulation.  For a simple letter matrix test run::

  python3 -m mindaffectBCI.examples.presentation.selectionMatrix

*NOTE*: For this type of rapid visual stimulation BCI, it is *very* important that the visual flicker be displayed *accurately*.  However, as the graphics performance of computers varies widely it is hard to know in advance if a particular configuration is accurate enough.  To help with this we also provide a graphics performance checker, which will validate that your graphics system is correctly configured.  You can run this with::

  python3 -m mindaffectBCI.examples.presentation.framerate_check

As this runs  it will show in a window your current graphics frame-rate and, more importantly, the variability in the frame times.  For good BCI performance this jitter should be <1ms.  If you see jitter greater than this you should probably adjust your graphics card settings.  The most important setting to consider is to be sure that you  have `_vsync_ <https://en.wikipedia.org/wiki/Screen_tearing#Vertical_synchronization>` *turned-on*.  Many graphics cards turn this off by default, as it (in theory) gives higher frame rates for gaming.  However, for our system, frame-rate is less important than *exact*  timing, hence always turn vsync on for visual Brain-Compuber-Interfaces!

System Overview
---------------

The mindaffectBCI consists of 3 main pieces:

 - *decoder* : This piece runs on a compute module (the raspberry PI in the dev-kit), connects to the EEG amplifer and the presentation system, and runs the machine learning algorithms to decode a users intended output from the measured EEG.

 - *presentation* : This piece runs on the display (normally the developers laptop, or tablet)), connects to the decoder, and shows the user interface to the user,  with the possible flickering options to pick from.

 - *output* : This piece, normally runs on the same location as the  presentation, but may be somewhere else, and also connects to the decoder.  It listens from 'selections' from the decoder, which indicate that the decoder has decided the user want's to pick a particular option,  and makes that  selection happen -- for example by adding a letter to the current sentence, or moving a robot-arm,  or turning on or off a light.

The  detailed  system architeture of the mindaffecBCI is explained in more detail in `doc/Utopia _ Guide for Implementation of new Presentation and Output Components.pdf <https://github.com/mindaffect/pymindaffectBCI/blob/master/doc/Utopia%20_%20Guide%20for%20Implementation%20of%20new%20Presentation%20and%20Output%20components.pdf>`_, and is illustrated in this figure:

.. image:: https://github.com/mindaffect/pymindaffectBCI/blob/master/doc/SystemArchitecture.png


Simple *output* module
------------------------

An output module listens for selections from the mindaffect decoder and acts on them to create some output.  Here we show how to make a simple output module which print's "Hello World" when the presentation 'button' with ID=1 is selected.

Note: Note: this should be in a separate file from the *output* example above.  You can find the complete code for this minimal-presentation on our github `examples/output/minimal_output.py <https://github.com/mindaffect/pymindaffectBCI/blob/master/mindaffectBCI/examples/output/minimal_output.py>`_


.. code:: python

  # Import the utopia2output module
  from mindaffectBCI.utopia2output import Utopia2Output


Now we can create an utopia2output object and connect it to a running mindaffect BCI decoder. 

.. code:: python

  u2o=Utopia2Output()
  u2o.connect()


(Note: For this to succeed you must have a real or simulated mindaffectBCI decoder running somewhere on your network.)

Now we define a function to print hello-world

.. code:: python

  def helloworld(objID):
     print("hello world")


And connect it so it is run when the object with ID=1 is selected.


.. code:: python

  # set the objectID2Action dictionary to use our helloworld function if 1 is selected 
  u2o.objectID2Action={ 1:helloworld }


Finally, run the main loop

.. code:: python

  u2o.run()


For more complex output examples, and examples for controlling a `lego boost <https://www.lego.com/en-gb/themes/boost>`_ robot, or a `philips Hue <https://www2.meethue.com/en-us>`_ controllable light, look in the `examples\output` directory. 

Simple *presention* module
----------------------------

Presentation is inherently more complex that output as we must display the correct stimuli to the user with precise timing and communicate this timing information to the mindaffect decoder.  Further, for the BCI operation we need to operation in (at least),

- _calibration_ mode where we cue the user where to attend to obtain correctly labelled brain data to train the machine learning algorithms in the decoder and
- _prediction_ mode where the user actually uses the BCI to make selections.

The *noisetag* module mindaffectBCI SDK provides a number of tools to hide this complexity from the application developers.  Using the most extreeem of these all the application developer has to do is provide a function to _draw_ the display as instructed by the noisetag module.

Note: this should be in a separate file from the *output* example above.  You can find the complete code for this minimal-presentation on our `examples/presentation/minimal_presentation.py <https://github.com/mindaffect/pymindaffectBCI/blob/master/mindaffectBCI/examples/presentation/minimal_presentation.py>`_

To use this.  Import the module and creat the noisetag object.

.. code:: python

  from mindaffectBCI.noisetag import Noisetag
  nt = Noisetag()


Note\: Creation of the `Noisetag` object will also implictly create a connection to any running mindaffectBCI decoder - so you should have one running somewhere on your network.

Write a function to draw the screen.  Here we will use the python gaming librar [pyglet](www.pyglet.org) to draw 2 squares on the screen, with the given colors.


.. code:: python

  import pyglet
  # make a default window, with fixed size for simplicty
  window=pyglet.window.Window(width=640,height=480)

  # define a simple 2-squares drawing function
  def draw_squares(col1,col2):
    # draw square 1: @100,190 , width=100, height=100
    x=100; y=190; w=100; h=100;
    pyglet.graphics.draw(4,pyglet.gl.GL_QUADS,
                         ('v2f',(x,y,x+w,y,x+w,y+h,x,y+h)),
			                   ('c3f',(col1)*4))
    # draw square 2: @440,100
    x=640-100-100
    pyglet.graphics.draw(4,pyglet.gl.GL_QUADS,
                         ('v2f',(x,y,x+w,y,x+w,y+h,x,y+h)),
			                   ('c3f',(col2)*4))    


Now, we need a bit of python hacking.  Because our BCI depends on accurate timelock of the brain data (EEG) with the visual display, we need to have accurate time-stamps for when the display changes.  Fortunately, pyglet allows us to get this accuracy as it provides a `flip` method on windows which blocks until the display is actually updated.  Thus we can use this to generate accurate time-stamps.   We do this by adding a time-stamp recording function to the windows normal `flip` method with the following magic:

.. code:: python

  # override window's flip method to record the exact *time* the
  # flip happended
  def timedflip(self):
    '''pseudo method type which records the timestamp for window flips'''
    type(self).flip(self) # call the 'real' flip method...
    self.lastfliptime=nt.getTimeStamp()
  import types
  window.flip = types.MethodType(timedflip,window)
  # ensure the field is already there.
  window.lastfliptime=nt.getTimeStamp()
	  
					   
Now we write a function which,
1) asks the `noisetag` framework how the selectable squares should look,
2) updates the `noisetag` framework with information about how the display was updated.


.. code:: python

  # dictionary mapping from stimulus-state to colors
  state2color={0:(.2,.2,.2), # off=grey
               1:(1,1,1),    # on=white
               2:(0,1,0),    # cue=green
  	       3:(0,0,1)}    # feedback=blue
  def draw(dt):
    # send info on the *previous* stimulus state.
    # N.B. we do it here as draw is called as soon as the vsync happens
    nt.sendStimulusState(timestamp=window.lastfliptime)
    # update and get the new stimulus state to display
    # N.B. update raises StopIteration when noisetag sequence has finished
    try : 
        nt.updateStimulusState()
        stimulus_state,target_state,objIDs,sendEvents=nt.getStimulusState()
    except StopIteration :
        pyglet.app.exit() # terminate app when noisetag is done
        return
    # draw the display with the instructed colors
    # draw the display with the instructed colors
    if stimulus_state : 
        draw_squares(state2color[stimulus_state[0]],
                     state2color[stimulus_state[1]])


As a final step we can attached a **selection** callback which will be called whenever a selection is made by the BCI.

.. code:: python

  # define a trival selection handler
  def selectionHandler(objID):
    print("Selected: %d"%(objID))    
  nt.addSelectionHandler(selectionHandler)

Finally, we tell the `noisetag` module to run a complete BCI 'experiment' with calibration and feedback mode, and start the `pyglet` main loop.


.. code:: python

  # tell the noisetag framework to run a full : calibrate->prediction sequence
  nt.setnumActiveObjIDs(2)  # say that we have 2 objects flickering
  nt.startExpt(nCal=10,nPred=10)
  # run the pyglet main loop
  pyglet.clock.schedule(draw)
  pyglet.app.run()

This will then run a full BCI with 10 *cued* calibration trials, and uncued prediction trials.   During the calibration trials a square turning green shows this is the cued direction.  During the prediction phase a square turning blue shows the selection by the BCI.

For more complex presentation examples, including a full 6x6 character typing keyboard, and a color-wheel for controlling a `philips Hue light <https://www2.meethue.com/en-us>`_ see the `examples/presentation` directory.
