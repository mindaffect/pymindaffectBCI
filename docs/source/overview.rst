Overview
========

The `mindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>`_ is an open-source `brain-computer-interface <https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface>`_ framework, aimed at enabling users to easily develop new ways for interacting with their computers directly using their brains.

What makes mindaffectBCI different
----------------------------------

There are already a number excellent of open-source BCI frameworks out there, such as `bci2000 <https://www.bci2000.org>`_ and `openVibe <http://openvibe.inria.fr/>`_, with large active user communities.   So why develop another one?

When we looked at the existing frameworks, we noticed that whilst they were great and extremely flexible, they weren't so easy to use for developing end-user applications.   So they didn't quite meet our objective of making it easy to develop new modes of interaction using brain signals.  

Our aim is to simplify or hide the brain-signals aspect of the BCI as much as possible (or wanted) and so allow developers to focus on their applications.  With this in mind mindaffectBCI is designed to be;

  * **modular** 
  * **cross-platform** -- run on all major desktop OSs (talk to `us <info@mindaffect.nl>`_ if you are interested in mobile)
  * **hardware neutral** -- support (via `brainflow <https://github.com/OpenBCI/brainflow>`_) many existing amplifiers, with instructions to easily add more,
  * **language neutral** -- provide APIs and examples for; python, java, c/c++, c#, swift
  * **batteries included** -- out of the box include a high-performance BCI, and examples for using this with common app development frameworks; (python, unity, swift) 

Target users of mindaffectBCI
-----------------------------

User Interface Designers
++++++++++++++++++++++++

The main target users for the mindaffectBCI are Application Developers who would like to add brain control to their applications, for example to add brain controls to a smart TV application.  Within that we provide tools for particular user groups.

 * Game Designers:  Do you want to add brain controls to an existing game?  Or make a new game including Brain controls as a novel interaction modality?  You can easily do this, in a cross-platform way, using our `unity <https://unity.com>`_ plugin available `here <https://github.com/mindaffect/unitymindaffectBCI>`_.

 * Patient Technical Support Teams: One of the key motivators behind the MindAffect team is to make BCIs available to improve peoples lives.  We can help some patients directly `ourselves <https://youtu.be/cCmHZb5vfso?t=47>`_, but cannot support every possible patient and their environment.  Instead, we try to provide the tools so patient support teams can themselves fit the BCI to their patients needs.  For this, we provide a basic text communication application out-of-the-box, with guidance on how to customise this for their users needs, for example different alphabets, or control of novel output devices.  
 
 * Hackers and Makers: Do you want to add brain control to your raspberry-pi robot, Lego robot, sphero or drone?  Now you can, either by using a simple control app on your laptop, or (more fun) by adding LEDs or `LASERS(!!!) <https://youtu.be/brN0YOg1AvY>`_ to your robot for direct control.  We provide examples for driving LEDs from a raspberry Pi, and are happy to help using other hacker boards (micro:bit) or even the LEDs on your drone. 

Neuroscience Students and Researchers
+++++++++++++++++++++++++++++++++++++

For the user interface designers, we deliberately hide the brain-signals as much as possible.  However, a BCI also provides an excellent tool for learning about basic neuroscience -- in particular how the brain responses to external stimulus.   For these users provide tools for the on-line real-time visualization of the stimulus specific responses.  Importantly, these visualizations utilize the same technology as the on-line BCI, which uses machine learning techniques to improve signal quality and separate the responses from overlapping brain responses.  This, gives students a clear view of the brain response in a short amount of time allowing for interactive learning and experimentation with stimulus parameters or mental strategies.  For example, so students can directly see the common (p300) and differential (perceptual) responses when using visual vs. auditory oddball paradigms.  


Machine Learning Engineers / Data Scientists
+++++++++++++++++++++++++++++++++++++++++++

Modern BCIs (including our own) rely heavily on machine learning techniques to process the noisy data gathered from EEG sensors and cope with the high degree of variability in responses over different individuals and locations.  MindAffect firmly believes that with more sophisticated machine learning techniques more useful information can be extracted from even 'low quality' consumer grade EEG data.  What is really needed is a combination of more and larger datasets on which to train the algorithms and better techniques tuned to the specific issues of neural data.  The mindaffect BCI aims to facilitate this data lead approach to BCI in two ways. 

 * Firstly, by making it easier to rapidly gather relatively large EEG datasets by using consumer grade EEG devices and applications designed in your prefered application development framework.  For example, by using a raspberry Pi, headphones, and EEG headband and an openBCI ganglion to measure the brain's response to different music types.

 * Secondly, by providing a `sklearn <scikit-learn.org>`_ compatiable interface for machine learning developers to experiment with different learning algorithms.  To help you get started quickly we provide `example data <https://www.kaggle.com/mindaffect/mindaffectbci>`_ and `analysis notebooks <https://www.kaggle.com/mindaffect/mindaffectbci/notebooks>`_ so you can experiment even without any EEG hardware. 

Project ideas for users
-----------------------

There are lots of things you could do with the mindaffectBCI, see `here <https://mindaffect-bci.readthedocs.io/en/latest/project_ideas.html>`_ for some of our own projects.  Below is a list of ideas to get you inspired!

1) Brain controlled robot arm - use a laser or projector to illuminate objects to move, e.g. chess pieces, or food to eat, and the BCI to select which piece to move and where too.  Lazy chess or snacking. See `here <https://youtu.be/brN0YOg1AvY>`_ for an example.

2) Neural `shazam <https://www.shazam.com/>`_ or Perceived music detection - Identify what music someone is listening to directly from their brain response. 

3) Tactile BCI - Allow someone to answer yes-no-questions (or even spell words) by concentrating on different parts of their body.

4) Brain Defenders -- Play `missile-command <https://en.wikipedia.org/wiki/Missile_Command>`_ using only your brain to pick where to send your defending missiles.  Or go further and do it in `Virtual Reality <https://youtu.be/kKdPnhxWhow>`_

5) Brain home-automation - Use brain control to change the color of your lights, like `Philips Hue control <https://youtu.be/6Vppourxiiw>`_, or to control your TV.

6) Real-world telekinesis -  Use your brain to shoot storm-troopers in a modern tin-can-alley, like `this <https://youtu.be/MsWDKX7Bqbs>`_

7) Brain-Golf (or Croquet)-- play golf with your brain by controlling a `sphero <https://sphero.com/>`_ from a tablet.  See `Sphero control <https://youtu.be/0Bu0caBzeDw>`_ for some inspiration.

8) Brain control of your phone?  Use our unity or iOS APIs to build a phone app controllable with your brain?Like `this <https://youtu.be/1BB0kgKJ0_w>`_

9) Make physical 'though buttons' -  Use a cheap IoT computer, such as `micro:bit <https://microbit.org/>`_ or `EPS32 <https://en.wikipedia.org/wiki/ESP32>`_ to present the BCI stimuli and directly control a device.

10) Brain Controlled Tele-presence - Use a simple Robot with a video-camera, such as this `pi-bot <https://www.instructables.com/Raspberry-Pi-Wifi-Video-Streaming-Robot/>`_, with an application on your laptop to drive the robot with brain signals.  

11) Add a `MQTT <https://mqtt.org/>`_ output interface -- MQTT is a widely used protocol for controlling IoT devices, add an output module to map BCI selections into MQTT commands to easily control a huge range of IoT devices. 

12) Bluetooth Low-Energy (BLE) output interface -- add an output module to map BCI selections into BLE characteristics, see the message-spec for the targeted UUIDs `here https://mindaffect-bci.readthedocs.io/en/latest/MessageSpec.html#service-selection>`_, and then make BLE controlled applications, for example on a `micro:bit <https://microbit.org/>`_.

13) Continuous control -- the base mindaffectBCI is designed around 'trials' or 'selections' where a single discrete button is pressed after a certain amount of time.  Internally, it does this by computing a prediction roughly every .1s -- can these intermediate controls be used for finer grained control applications, e.g. cursor control?