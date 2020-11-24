## Add a new amplifier to mindaffectBCI

Out of the box mindaffectBCI supports a large range of amplifiers, either via. it's use of `brainflow <brainflow.org>`_ or with amplifier drivers developed by MindAffect.  But what if you have a new wizzy amplifier and it's not currently supported by brainflow? How can you easily add support for this cool new device to the mindaffectBCI -- ideally without introducing a lot of extra dependencies in your code?

In this tutorial you will learn:
 1. The low-level format used to stream to the mindaffectBCI hub
 2. The importance of **device-level** time-stamps in ensuring the data you send it as good as possible for BCI applications
 3. How to write a simple 'fake-data' simulated amplifier stream in C, python, Java or C#

###The mindaffectBCI DATAPACKET transmission format

Data is sent from the amplifier to the mindaffectBCI in packets containing an array of channels by samples in 32-bit floats (were channels vary fastest).  At the lowest / simplest level, all an amplifier has to do is:
  1. Open a TCP network socket to connect to the hub on port 8400
  2. Send a stream of DATAPACKET to the hub

Note: At it's lowest level this is a **one-way** data stream -- so your amp driver logic can be extreemly simple.  Further, you do not **need** to provide any meta-information about the amplifier, e.g. model, a sample rate, or channel names etc., for it to work (though it is nice if you can send this information in a DATAHEADER packet.)  Thus, quickly developing a new amp driver for testing can be extreemly simple -- as you will see below it's literially ~6 lines of pure-python.

The detailed format of the DATAPACKET messages (along with all the other message types used by the mindaffectBCI) are given in the system `message specification <https://mindaffect-bci.readthedocs.io/en/latest/MessageSpec.html>`_.  The core section for the DATAPACKET messages is repeated here for clarity. 

.. raw:: html

  <table>
    <tr>
     <td>Name: 
  <h6 id="datapacket">DATAPACKET</h6>


     </td>
     <td>UID: “D”
     </td>
    </tr>
    <tr>
     <td><strong>Sender:</strong> 
  <p>
  Acquisation Device  (e.g. EEG)
     </td>
     <td><strong>Receiver:</strong> 
  <p>
  Recogniser
     </td>
    </tr>
    <tr>
     <td colspan="2" ><strong>Purpose:</strong> 
  <p>
  Send raw data as measured by the acquisition device to the decoder.
     </td>
    </tr>
    <tr>
     <td colspan="2" ><strong>Format:</strong> 
  <p>
  Basically this is not something we can specify as it depends on the exact hardware device.  Minimum spec for us:

  <table style="width: 100%">
      <colgroup>
         <col span="1" style="width: 10%;">
         <col span="1" style="width: 25%;">
         <col span="1" style="width: 65%;">
      </colgroup>

    <thead>
    <tr>
     <td><strong>Slot</strong>
     </td>
     <td><strong>Type (= value)</strong>
     </td>
     <td><strong>Comment</strong>
     </td>
    </tr>
    </thead>
    <tr>
     <td>UID
     </td>
     <td>1 of char = “D” 
     </td>
     <td>Message UID
     </td>
    </tr>
    <tr>
     <td>version
     </td>
     <td>1 of uint8
     </td>
     <td>Message version number (0)
     </td>
    </tr>
    <tr>
     <td>length
     </td>
     <td>[1] of uint16 (short)
     </td>
     <td>Total length of the remaining message in bytes.
     </td>
    </tr>
    <tr>
     <td>timestamp
     </td>
     <td>[1] of int32
     </td>
     <td>Time of the *first* sample of this data packet.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
     </td>
    </tr>
    <tr>
     <td>nsamples
     </td>
     <td>[1] of int 32
     </td>
     <td>The number of samples (i.e. time-points) in this datapacket (Note: the nchannels is infered to be (length-8)/nsamples/4)
     </td>
    </tr>
    <tr>
     <td>data
     </td>
     <td>[ nchannels x nSamp ] of single 
     </td>
     <td>The raw packed data
     </td>
    </tr>
  </table>

  Notes:

  32bit timestamps @1ms accuracy means the timestamps will wrap-around in 4294967296/1000/60/60/24  = 50 days.. Which is way more than we really need….  

  With 24 bits this would be 4hr..  For implementation simplicity standard 32bit ints are prefered.

     </td>
    </tr>
  </table>

Based on this format, in python given raw data in `samples` which is a (samples,channels) np.float32 numpy array and using the `struct` package you can make a valid datapacket with:

.. code::

    DP = struct.pack("<BBHii%df"%(samples.size),'D',0,2+4+samples.size*4,samples.shape[-1],samples.ravel())



