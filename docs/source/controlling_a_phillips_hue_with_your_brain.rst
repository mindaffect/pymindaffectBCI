VideoTutorial: Controlling a Phillips HUE with your brain
=======================================================

The Phillips `HUE <https://www.philips-hue.com/>`_ compute controlled lighting system.  In this video tutorial we show how you can easily control the color of these lights with brain control using our python framework and the great `phue <https://github.com/studioimaginaire/phue>`_ python library. 

.. raw:: html

    <div style="text-align: center; margin-bottom: 2em;">
    <iframe width="100%" height="350" src="https://www.youtube.com/embed/-vTBUkfIMA4" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
    </div>

**Note: this video is orginally for the `mindaffect Kickstarter Kit <https://www.kickstarter.com/projects/bci/make-100-create-your-own-brain-computer-interface>`_  so includes some references to the kit or fake-recogniser which are not relevant for the open-source release.  All the unity specific parts remain unchanged.  Just start the fake-recogniser with

.. code::

   python3 -m mindaffectBCI.online_bci --config_file fake_recogniser