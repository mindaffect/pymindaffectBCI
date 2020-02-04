mindaffectBCI
=============
This repository contains the python SDK code for the Brain Computer Interface (BCI) developed by the company [Mindaffect](https://mindaffect.nl).

File Structure
--------------
This repository is organized roughly as follows:

 - `mindaffectBCI` - contains the python package containing the mindaffectBCI SDK.  Important modules within this package are:
   - noisetag.py - This module contains the main API for developing User Interfaces with BCI control
   - utopiaController.py - This module contains the application level APIs for interacting with the MindAffect Decoder.
   - utopiaclient.py - This module contains the low-level networking functions for communicating with the MindAffect Decoder - which is normally a separate computer running the eeg analysis software.
   - stimseq.py -- This module contains the low-level functions for loading and codebooks - which define how the presented stimuli will look.
 - `codebooks` - Contains the most common noisetagging codebooks as text files
 - `examples` - contains python based examples for Presentation and Output parts of the BCI. Important sub-directories
   - output - Example output modules.  An output module translates BCI based selections into actions.
   - presentation - Example presentation modules.  A presentation module, presents the BCI stimulus to the user, and is normally the main UI.

Installing mindaffectBCI
------------------------

That's easy::

  pip3 install mindaffectBCI


Testing the mindaffectBCI SDK
-----------------------------

This SDK provides the functionality needed to add Brain Controls to your own applications.  However, it *does not* provide the actual brain measuring hardware (i.e. EEG) or the brain-signal decoding algorithms (they will be made available later in a mindaffect decoder repository). 

In order to allow you to develop and test your Brain Controlled applications without connecting to a real mindaffect Decoder, we provide a so called "fake recogniser".  This fake recogniser simulates the operation of the true mindaffect decoder to allow easy development and debugging.  Before starting with the example output and presentation modules you should start this fake recogniser by running, either ::

  bin/startFakeRecogniser.bat
  
if running on windows, or  ::

  bin/startFakeRecogniser.sh

if running on linux/macOS

If successfull, running these scripts should open a terminal window which shows the messages recieved/sent from your example application.

Note: The fakerecogniser is written in [java](https://www.java.com), so you will need a JVM with version >8 for it to run.  If needed download from [here](https://www.java.com/ES/download/)


Simple *output* module
------------------------

An output module listens for selections from the mindaffect decoder and acts on them to create some output.  Here we show how to make a simple output module which print's "Hello World" when the presentation 'button' with ID=1 is selected.


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

  def helloworld():
     print("hello world")


And connect it so it is run when the object with ID=1 is selected.


.. code:: python

  # set the objectID2Action dictionary to use our helloworld function if 1 is selected 
  u2o.objectID2Action={ 1:helloworld }


Finally, run the main loop

.. code:: python

  u2o.run()


For more complex output examples, and examples for controlling a [lego boost](https://www.lego.com/en-gb/themes/boost) robot, or a [philips Hue](https://www2.meethue.com/en-us) controllable light, look in the `examples\output` directory. 

Simple *presention* module
----------------------------

Presentation is inherently more complex that output as we must display the correct stimuli to the user with precise timing and communicate this timing information to the mindaffect decoder.  Further, for the BCI operation we need to operation in (at least),

- _calibration_ mode where we cue the user where to attend to obtain correctly labelled brain data to train the machine learning algorithms in the decoder and
- _prediction_ mode where the user actually uses the BCI to make selections.

The *noisetag* module mindaffectBCI SDK provides a number of tools to hide this complexity from the application developers.  Using the most extreeem of these all the application developer has to do is provide a function to _draw_ the display as instructed by the noisetag module.

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
    nt.sendStimulusState()
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


Finally, we tell the `noisetag` module to run a complete BCI 'experiment' with calibration and feedback mode, and start the `pyglet` main loop.


.. code:: python

  # tell the noisetag framework to run a full : calibrate->prediction sequence
  nt.startExpt([1,2],nCal=10,nPred=10)
  # run the pyglet main loop
  pyglet.clock.schedule(draw)
  pyglet.app.run()

This will then run a full BCI with 10 *cued* calibration trials, and uncued prediction trials.   During the calibration trials a square turning green shows this is the cued direction.  During the prediction phase a square turning blue shows the selection by the BCI.


For more complex presentation examples, including a full 6x6 character typing keyboard, and a color-wheel for controlling a [philips Hue light](https://www2.meethue.com/en-us) see the `examples/presentation` directory.
