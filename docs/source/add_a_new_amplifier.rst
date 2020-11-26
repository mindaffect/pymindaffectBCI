Add a new amplifier to mindaffectBCI
====================================

Out of the box mindaffectBCI supports a large range of amplifiers, either via. it's use of `brainflow <brainflow.org>`_ or with amplifier drivers developed by MindAffect.  But what if you have a new wizzy amplifier and it's not currently supported by brainflow? How can you easily add support for this cool new device to the mindaffectBCI -- ideally without introducing a lot of extra dependencies in your code?

In this tutorial you will learn:
 1. The low-level format used to stream to the mindaffectBCI hub
 2. The importance of **device-level** time-stamps in ensuring the data you send it as good as possible for BCI applications
 3. How to write a simple 'fake-data' simulated amplifier stream in Python.

The mindaffectBCI DATAPACKET transmission format
------------------------------------------------

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
     <td>[ nchannels x nSamp ] of float32 
     </td>
     <td>The raw packed data
     </td>
    </tr>
  </table>

  Notes: 32bit timestamps @1ms accuracy means the timestamps will wrap-around in 4294967296/1000/60/60/24  ~= 50 days.  

     </td>
    </tr>
  </table>


Based on this format, in python given an integer *timestamp*, raw data in *samples* which is a (samples,channels) `np.float32` numpy array and using the `struct` package, you can make a valid datapacket with::

    DP = struct.pack("<BBHii%df"%(samples.size),ord('D'),0,4+4+samples.size*4,timestamp,samples.shape[-1],*(s for s in samples.ravel()))

Note: This line uses some horrible python hacks; like: `ord('D')` to convert char->integer, `samples.ravel()` to convert the n-d samples to a 1-d matrix, `(s for s in samples.ravel())` to convert the nd-array to a python tuple, and the finally `*(...)` to expand the tuple into a set of arguments.

Minimal Acquisation Driver : Python
-----------------------------------

**Note:** this example designed for exposition purposes, implementators are better adviced to use the `utopiaclient.py` API, as it provides a more complete interface, with e.g. auto-discovery, error-recovery, two-way communication, and access to the full message vocabularly. 

To make the absolute minimum `fake-data` streamer we need to do 5 things:
 1. Open a TCP socket to connect to the hub.::
 
     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
     sock.open('localhost',8400)
 
 2. Get the fake-data packet::
 
     n_ch = 4
     n_samples = 10
     samples = np.random.standard_normal((n_ch,n_samples)).astype(np.float32)
 
 3. Get the current time-stamp::
 
     timestamp = int(time.perf_counter()*1000) % (1<<31) # N.B. MUST fit in 32bit int
     
 4. Make the DATAPACKET::
 
     DP = struct.pack("<BBHii%df"%(samples.size),ord('D'),0,4+4+samples.size*4,timestamp,samples.shape[-1],*(s for s in samples.ravel()))
 
 5. send the message::
 
     sock.send(DP)

 Or to wrap it all up into a single 10-line code block (without imports), with a loop to stream for-ever, and a sleep to rate-limit to a desired effective sample rate::

     import numpy as np
     import time
     import socket
     import struct

     def fakedata_stream(host='localhost', sample_rate=100, n_ch=4, packet_samples=10):
         inter_packet_interval = packet_samples / sample_rate

         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
         sock.connect((host,8400))

         while True:
             samples = np.random.standard_normal((n_ch,packet_samples)).astype(np.float32)
             timestamp = int(time.perf_counter()*1000) % (1<<31) # N.B. MUST fit in 32bit int
             DP = struct.pack("<BBHii%df"%(samples.size),ord('D'),0,4+4+samples.size*4,timestamp,samples.shape[-1],*(s for s in samples.ravel()))
             sock.send(DP)
             time.sleep(inter_packet_interval) # sleep to rate limit to sample_rate Hz

Congratulations, you have just written your own custom datapacket streamer for the mindaffect BCI.   

To adapt this to use data from an actual hardware device, then simply replace the `samples = np.random.standard_normal...` line with a call to the hardware function which gets the actual samples from the amplifier. 

The Importance of **Amplifier** timestamps
------------------------------------------

At it's core any evoked-response BCI (like the mindaffect BCI) must align at least two data-streams, namely the EEG stream (from the amplifier) and the STIMULUS stream (from the presentation device).  Doing this alignment with high latency links (such as wireless network connections) can be a complex problem.  The solution used in the mindaffect BCI is to use a **local** clock on the device (i.e. amplifier, screen) to attach accurate **timestamps** to the data at source, and then use a jitter rejecting and step detection algorithm in the decoder to align the time-stamp streams (which due to electronic issues can have different offsets and may drift relative to each other) to the common decoder clock.  

What this means for amplifier implementors is that **it is very important** to time-stamp your data as close to the source as possible.  We have found that using the poor quality clocks in a cheap devices is a better time-stamp source than an high quality clock in a PC -- basically because even a poor quality device clock has a sub-millisecond jitter and only drifts by approx 1 millisecond / second, whereas wireless transmission jitter can be 10 to 100 milliseconds / second with a similar 1ms/s drift.  When coupled to potential sample loss in transmission, this makes 'recieve-time' timestamps a poor subistute for 'measurment-time' device-level timestamps. 

Summary
-------

Adding a new amplifier to the mindaffect BCI can be done by either:
  1. Adding the new amplifier to brainflow
  2. Streaming the data on a TCP socket in the timestamped DATAPACKET format



