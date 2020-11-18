## mindaffectBCI : Message Specification


### Purpose

This document describes in the messages which are passed between the different components of the [mindaffect BCI](https://github.com/mindaffect/pymindaffectBCI) system, and the low-level byte-struture used for the messages.  The message specification is *transport agnostic* in that the messages themselves may be sent over different ‘wire-protocols’, such as BTLE, UDP, TCP, etc.

Note: This specification is intended for developing new low-level components to interface to the mindaffectBCI.  Most developers / users can directly use one of the provided higher-level APIs, such as [python](https://github.com/mindaffect/pymindaffectBCI), [java](https://github.com/mindaffect/javamindaffectBCI), [c#/unity](https://github.com/mindaffect/unitymindaffectBCI).


### Objectives



*   Stateless: as much as possible messages are self-contained information updates such that the interpretation of a message depends minimally on the reception of other messages.  In this way, we are both robust to loss of a single message and failure/rebooting of an individual component, and also simplify the later loading/replay of saved data as we can basically start playback at any point.  The disadvantage of this is however the messages tend to be longer than needed due to the additional redundancy.
*   Simple + Readable: as much as possible the message spec will be human readable, such that for example message types are encoded in plain strings.  However, the spec will also be simple to read manipulate the messages, thus integers will be sent as integers, and arrays as packed binary arrays. 
*   Efficient: as much as possible (without violating the stateless and readable objectives) the message spec will be efficient in space usage in transmission.
*   Compact : some of our transport layers have very small payload sizes (e.g. BLE has 20 bytes) thus the messages should be compact such that a useful stateless message can be sent in this payload size.
*   Latency  tolerant : we cannot guarantee timely transmission of messages between components.  Thus, the spec will (where appropriate) include additional time-stamp information to allow a ‘true’ message time-line to be reconstructed.


### Structure

The message specification is structured as follows:



*   Message Name: a human readable informative name for the message
*   Message UID : ascii character used to uniquely identify this message.  This is required to be the first character of any message.
*   Sender: the component which produces the message
*   Receiver: the component which receives the message
*   Purpose: the reason for sending this message between these two components
*   Format: the detailed structure of the message in terms of the basic types (i.e. char, int8, int16, uint8, single, double etc.) it is made up from.  Each slot of the format has:
    *   Name: 
    *   Type Information: 
    *   Comment: human readable description of the purpose of this slot


### **Message Specifications** 


### Endianness 

To simplify things we *require* that all numbers be encoded in **LITTLE ENDIAN**.


### Message Header 

All messages start with a standard header consisting of:

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
   <td>1 of char 
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>1 of uint8 (0)
   </td>
   <td>Message version number (0)
   </td>
  </tr>
  <tr>
   <td>length
   </td>
   <td>[1] of uint16 (short)
   </td>
   <td>Total length of the remaining message in bytes.  This can be used to allow *forward compatibility* as clients can skip gracefully skip messages where they do not understand the payload, e.g. because of an unknown messageUID+version combination.
   </td>
  </tr>
  <tr>
   <td>payload
   </td>
   <td>[ length ] of byte 
   </td>
   <td>The message payload, i.e. the rest of the message
   </td>
  </tr>
</table>



### General Utility Messages


<table>
  <tr>
   <td>Name:
<h6> HEARTBEAT</h6>


   </td>
   <td><strong>UID: “H”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>ANY component</strong>
<p>
<strong>REQUIRED: STIMULUS</strong>
<p>
<strong>REQUIRED: RECOGNISER</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>ANY component</strong>
<p>
<strong>REQUIRED: RECOGNISER</strong>
<p>
<strong>REQUIRED: UI</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
These heartbeat messages have two main purposes:
<ol>

<li>As an <strong>I’m alive</strong> indication.  If the heartbeat stop then we can conclude the recieve component has crashed. 

<li>As a ‘<strong>clock alignment</strong>’ query.  They payload of the HEARTBEAT signal is the current timestamp of the receiver component.  Thus it can be used to track and align the sender’s clock with the recievers.

<p>
Notes:
<ul>

<li>The *exact* interval between HEARTBEATS is dependent on the client.  However the <strong>maximum</strong> interval is set to MAXHEARTBEATINTERVAL which is 4 seconds by default.

<li>It is <strong>required </strong>that <strong>all </strong>STIMULUS components send HEARTBEAT messages to RECOGNISER as soon as these two components establish an initial connection. 

<li>It is <strong>required </strong>that RECOGNISER sends HEARTBEAT messages to all clients as soon as these two components establish an initial connection.
</li>
</ul>
</li>
</ol>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “H”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time this signal quality measure was computed.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Name: 
<h6 id="subscribe">SUBSCRIBE</h6>


   </td>
   <td><strong>UID: “B”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>ANY component</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>RECOGNISER</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
These subscribe messages main purpose is to reduce the network and computation load on both the Utopia-Hub message server and the client by allowing the client to inform the hub about which messages the client is interested in.   
<p>
Notes:
<ul>

<li>As subscribe message may be sent at any time, to update the set of messages forwarded to this client from that point on.

<li>If no subscribe message is sent to the utopia-hub then by default the client receives all messages. 

<li>Even if a client attempts to subscribe to <strong>no </strong>messages, it will always receive HEARTBEAT messages.
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “B”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time this subscribe message was sent.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>subscriptionList
   </td>
   <td>[#messages] of char
   </td>
   <td>A string array of the message UIDs this client would like forwarded to it.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Name: 
<h6>LOG</h6>


   </td>
   <td><strong>UID: “L”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Any</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
General logging of the state of any individual component.  Currently this is used in the Recogniser to log things like it’s code version, where the data is being saved, and where the internal log-files are being saved.
<p>
Note:
<ul>

<li>This message is <strong>not </strong>a general debug logging system, in particular it is not designed for high-performance logging, but a more general logging ability to send general system information between components, and in particular to any UI component.  For debug logging we suggest you save to local storage on the client device and only log information on where this debug-log can be found for later retrevial.
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “L”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time the mode change message was sent.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>message
   </td>
   <td>[1] of string
   </td>
   <td>String with the logging information.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Presentation -> Recogniser 


<table>
  <tr>
   <td>Name: 
<h6 id="stimulusevent">STIMULUSEVENT</h6>


   </td>
   <td><strong>UID: “E”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Stimulus</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Provide the decoder with updated information about the current stimulus state.
<p>
Note this message format is designed to allow for *stateless* updates with very small message sizes.  Thus, each message is a self-contained report of (part of) the stimulus state at a given time, allowing to cut simulus-state updates over multiple messages without having to worry about message sequence numbers etc.
<p>
Note: If we are **really** pressed for message sizes then we can define a less general but more compact STIMULUSEVENT message based on compressing the STIMULUSSTATE structure into a single ‘byte’ with 7-bits for objectUID and 1-bit for state.  However, I think the implementation simplicity of the current message format makes this prefered even with the message size overhead.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

<table style="width: 100%">
    <colgroup>
       <col span="1" style="width: 10%;">
       <col span="1" style="width: 25%;">
       <col span="1" style="width: 5%;">
       <col span="1" style="width: 60%;">
    </colgroup>

  <thead>
  <tr>
   <td><strong>Slot</strong>
   </td>
   <td><strong>Type (= value)</strong>
   </td>
   <td><strong>length</strong>
   </td>
   <td><strong>Comment</strong>
   </td>
  </tr>
  </thead>
  <tr>
   <td>UID
   </td>
   <td>[1] of char = “E” 
   </td>
   <td>1 (1)
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>1 of uint8 = 0
   </td>
   <td>1 (2)
   </td>
   <td>Message version number (0)
   </td>
  </tr>
  <tr>
   <td>length
   </td>
   <td>[1] of uint16 (short)
   </td>
   <td>2 (4)
   </td>
   <td>Total length of the remaining message in bytes.
   </td>
  </tr>
  <tr>
   <td>timestamp
   </td>
   <td>[1] of uint32
   </td>
   <td>4 (8)
   </td>
   <td>Time of the *first* sample of this data packet.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>nobjects
   </td>
   <td>[1] of uint8
   </td>
   <td>1 (9)
   </td>
   <td>Number of objects with stimulus information *<strong>contained in this message*</strong>
   </td>
  </tr>
  <tr>
   <td>stimulusdict
   </td>
   <td>[nobjects] of <strong>STIMULUSDICT</strong>
   </td>
   <td>nobjects*2
<p>
 (nobjects*2+9)
   </td>
   <td>A dictionary of stimulus state information for nobjects.  The format of a single stimulus state entry is given below
   </td>
  </tr>
</table>


**STIMULUSDICT **structure


<table style="width: 100%">
    <colgroup>
       <col span="1" style="width: 10%;">
       <col span="1" style="width: 25%;">
       <col span="1" style="width: 5%;">
       <col span="1" style="width: 60%;">
    </colgroup>

  <thead>
  <tr>
   <td><strong>Slot</strong>
   </td>
   <td><strong>Type (= value)</strong>
   </td>
   <td><strong>length</strong>
   </td>
   <td><strong>Comment</strong>
   </td>
  </tr>
  </thead>
  <tr>
   <td>objectUID
   </td>
   <td>[1] of uint8
   </td>
   <td>1 (1)
   </td>
   <td>A unique identifier for object for which the stimulus state is being reported.
<p>
N.B. objectUID=0 is <strong>reserved </strong>for the <strong>true-target </strong>object when in supervised training mode.
   </td>
  </tr>
  <tr>
   <td>stimulusstate
   </td>
   <td>[1] of uint8
   </td>
   <td>1 (2)
   </td>
   <td>The updated stimulus state for the object with objectUID. The stimulus state indicates a relevant characteristic of the stimulus used by the decoder.  For example if this object is in a high or low brightness state.
   </td>
  </tr>
</table>


NOTES:



*   **objectID **- an object UID is (as the name implies) a *****unique*** **identifier for a particular stimulus object.  This is used ****both** **for indication of the stimulus state of an object **and **for indication of the identified target object when predictions are generated. 

    Object IDs for which *no* stimulus state information has been provided since the last decoder reset command are assumed to not be stimulated.  Object IDs do *not* have to be consecutive, and can be allocated arbitarly by the STIMULUS component 

*   **Stimulusstate **- The **encoding** used by the stimulus state is flexible in this version of the spec.  By convention stimulus state is treated as a **single **continuous level of the stimulus intensity, e.g. a grey-scale value.  However other encoding formats as possible without changing the message spec.  Example encodings could be; bit 0=long, bit(1)=short, or bits 0-4 for intensity and bits5-8 for color.   **It is the responsibility of CONFIG to ensure that STIMULUS and RECOGNISER agree on the interperation of the stimulus state object. **
*   Example Bandwidth Requirements: for a 36 output display with messages packed into 20bytes (as in BLE).  We have 6 bytes header overhead, leaving 7 objects in the message packet.  Thus we require 6 packets for a update on all objects in the display, i.e. 6*20 = 120 bytes / display update.  Thus @ 60 display rate we require: 120*60 = 7200 bytes/sec.  The spec for BLE gives, an *application* data rate of .27Mbit/sec = 270000 bit/sec = 33750 byte/sec.  Thus we need about 25% of the total available BLE bandwidth for this common use-case.
   </td>
  </tr>
</table>




### Recogniser -> Selection or Output 


<table>
  <tr>
   <td>Name: 
<h6 id="predictedtargetprob">PREDICTEDTARGETPROB</h6>


   </td>
   <td><strong>UID: “P”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Recogniser</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Output</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Inform the output component of the current predicted target and it’s probability of being correct so the output decide: a) if the prediction is confident enough, b) if so to generate the appropriate output.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “P”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time the prediction was generated.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>objectID
   </td>
   <td>[1] of uint8
   </td>
   <td>The objectID (as in the STIMULUSSTATE message) of the predicted target
   </td>
  </tr>
  <tr>
   <td>errrorprobability
   </td>
   <td>[1] of single
   </td>
   <td>The probability that objectID is **<strong>not** </strong>the true target object.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Name: 
<h6 id="predictedtargetdist">PREDICTEDTARGETDIST</h6>


   </td>
   <td><strong>UID: “F”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Recogniser</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Output</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Inform the output component of the current distribution over all possible predicted targets --to allow more fine grained decision making w.r.t. when and what output to generate.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “D”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time prediction was generated.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>targeterrordist
   </td>
   <td>[nObjects] of TARGETERRORDIST
   </td>
   <td>Dictionary of the error probabilities for all of the output objects known to the Recoginiser
   </td>
  </tr>
</table>


TARGETERRORDIST


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
   <td>objectUID
   </td>
   <td>[1] of uint8
   </td>
   <td>The objectID (as in the STIMULUSSTATE message) of the predicted target
   </td>
  </tr>
  <tr>
   <td>errorprobabililty
   </td>
   <td>[1] of single
   </td>
   <td>Error probably for this object
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Controller or Output  -> All


<table>
  <tr>
   <td>Name: 
<h6 id="modechange">MODECHANGE</h6>


   </td>
   <td><strong>UID: “M”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Controller</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Tell the decoder to switch to a new operating mode, e.g. switch from calibration to testing.
<p>
Currently, we have the following operating modes for the decoder:
<ul>

<li>Calibration.supervised - calibration with user target instruction.

<li>Calibration.unsupervised - calibration without user target instruction, a.k.a. zerotrain

<li>Prediction.static - generate predictions with a fixed model 

<li>Prediction.adaptive - generate predictions with adaptive regularisation

<li>NonStopLearning - run in non-stop learning mode
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “M”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time the mode change message was sent.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>newmode
   </td>
   <td>[1] of string
   </td>
   <td>String with the new decoder mode to enter.  One-of:
<p>
 Calibration.supervised,  Calibration.unsupervised, 
<p>
Prediction.static, 
<p>
Prediction.adaptive, 
<p>
ElectrodeQuality
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Name: 
<h6 id="newtarget">NEWTARGET</h6>


   </td>
   <td><strong>UID: “N”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Controller</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Tell the decoder that the user has switched to attempt selection of a new target.  For example, in the speller because the OUTPUT has made a character selection and the user is moving on to the next letter.  The RECOGNISER is expected to use this message to clear it’s prediction history and start fresh on a new output.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “N” 
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time the the new target change happened.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Name: 
<h6 id="selection">SELECTION</h6>


   </td>
   <td><strong>UID: “S”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Controller</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Tell other clients that a selection has been made (and possibly trigger output to be generated).  This further implies that  the user has switched to attempt selection of a new target (thus an explicit newtarget message is not required).  For example, in the speller because the OUTPUT/SELECTION has made a character selection and the user is moving on to the next letter.  
<p>
The RECOGNISER is expected to use this message to clear it’s prediction history and start fresh on a new output.  
<p>
An OUTPUT module may use this message to decide if it should perform it’s tasked output if the selection matches it’s trigger criteria, e.g. changing the TV channel in a TV remote. 
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “S” 
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time the the new target change happened.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>objID
   </td>
   <td>[1] of uint8
   </td>
   <td>Selected objectID. This objID should match that used in stimulusEvents
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Name: 
<h6 id="reset">RESET</h6>


   </td>
   <td><strong>UID: “R”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Controller</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Reset the RECOGNISER to a ‘fresh-start’ state, i.e. as if it has just been started with no saved information about training data or predictions.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “R” 
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
   <td>[1] of uint32
   </td>
   <td>Time the the reset happened.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Recogniser -> User Interface 


<table>
  <tr>
   <td><strong>Name: </strong>
<h6 id="signalquality">SIGNALQUALITY</h6>


   </td>
   <td><strong>UID: “Q”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Recogniser</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>UI</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Inform user about the quality of the electrode fit, so they can adjust the electrode positioning or contact to improve the connection to the scalp as much as possible.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “Q”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time this signal quality measure was computed.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>signalquality
   </td>
   <td>[nChan] of single
   </td>
   <td>A numeric measure in the range 0-1 of the channel noise to signal quality, where:
<p>
  0 -> a perfect electrode connection
<p>
  1 -> a completely bad electrode connection.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Acquisition -> Recogniser


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



<table>
  <tr>
   <td>Name: 
<h6>DATAHEADER</h6>


   </td>
   <td><strong>UID:</strong> A
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong> 
<p>
Acquisation Device (e.g. EEG)
   </td>
   <td><strong>Receiver:</strong> 
<p>
Recogniser
   </td>
  </tr>
  <tr>
   <td colspan="2" ><strong>Purpose:</strong>
<p>
Provide meta-information about the data provided by the acquisation device.  As a minimum this should be the number of channels sent and their sample rate.  Optionally should include channel location information.
   </td>
  </tr>
  <tr>
   <td colspan="2" ><strong>Format:</strong> 
<p>
Basically this is not something we can specify as it depends on the exact hardware device.  But suggestion is:

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
   <td>1 of char = “”
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
   <td>sample_rate
   </td>
   <td>1 of single
   </td>
   <td>Sampling rate of measurements
   </td>
  </tr>
  <tr>
   <td>nChan
   </td>
   <td>1 of int32
   </td>
   <td>Number of channels
   </td>
  </tr>
  <tr>
   <td>labels
   </td>
   <td>string
   </td>
   <td>Comma separated list of the textual names of the channels. E.g. “C3,C4,C5”
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Extension Messages

In this section are listed messages which may be useful in future, but will not be implemented in the V1.0 version of the system.


### Config -> Recogniser


<table>
  <tr>
   <td>Name:  
<h6 id="configudecoder">CONFIGURECOGNISER</h6>


   </td>
   <td><strong>UID: “C”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Config</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>Recogniser</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Update the general configuration of the recognizer, e.g. the response length.
<p>
   
<p>
Note: The side-effects of changing the configuration on the RECOGNISER is <strong>UNDEFINED.  </strong>In the worst case this may result in a complete restart of the RECOGNISER clearing all previous state, i.e. removing the trained classifier etc.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:
<p>
This message payload is a dictionary of name-value pairs encoded in JSON format of the configuration parameters that the RECOGNISER needs.
<p>
Note: This message could potentially be *very large*, and need to extend over multiple packets.  It is assumed the underlying transport will deal with this effectively.

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
   <td>[1] of char = “C”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time the the new target change happened.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>configJSONdict
   </td>
   <td>[1] &lt;string>
   </td>
   <td>This is the configuration encoded as a JSON dictionary string, e.g. “{ responseLength : 100 }”
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Other 


<table>
  <tr>
   <td><strong>Name: </strong>
<h6 id="ticktock">TICKTOCK</h6>


   </td>
   <td><strong>UID: “T”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>ANY component</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>ANY component</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
This message tells the receiver what the current time-stamp from the sender is. It also optionally represents a *query* to the receiver to send back it’s time-stamp as rapidly as possible. These messages have two main purposes:
<ol>

<li>To ‘clock alignment’ distribution method. Where the sender can inform different receivers of it’s current clock state. 

<li>As a ‘<strong>clock alignment</strong>’ query.  (Optionally) the receiver may response to this message by replying with the original query + it’s local time-stamp information.  This can then be used to estimate the message latency to more accurately align the different components clocks. 
</li>
</ol>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “T”
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
   <td>[1] of uint32
   </td>
   <td>Time this signal quality measure was computed.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>yourClock
   </td>
   <td>[1] of uint32
   </td>
   <td>(Optional) for a ticktock response message, the timestamp of the orginal TICKTOCK message we are responding to.
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong>Name: </strong>
<h6 id="currentmodel">CURRENTMODEL</h6>


   </td>
   <td><strong>UID: TBD</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Recogniser</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>UI</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Inform user about the parameters of the current model.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “”
   </td>
   <td>Message UID
   </td>
  </tr>
  <tr>
   <td>version
   </td>
   <td>[1] of uint8 = 0
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
   <td>[1] of uint32
   </td>
   <td>Time this model was computed.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>sizeSpace
   </td>
   <td>[2] of uint32
   </td>
   <td>[nChan,#E] dimensions of the following spatial filter matrix
   </td>
  </tr>
  <tr>
   <td>spatialfilter
   </td>
   <td>[nChan x #E] of single
   </td>
   <td>The current model’s spatial filter matrix
   </td>
  </tr>
  <tr>
   <td>sizeTime
   </td>
   <td>[2] of int32
   </td>
   <td>[Tau, #E] dimensions of the following impulse response matrix
   </td>
  </tr>
  <tr>
   <td>impulseresponse
   </td>
   <td>[Tau x #E] of single
   </td>
   <td>The current model’s estimated impulse response
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>



### Bluetooth Low Energy

When using BLE to communicate we need to define Services and Charactersitics.  Services represent groups of functionality which is exposed to other devices, with Characteristics the values to be communicated.  Within this message specification a unique characteristic maps directly onto a message type, with groups of messages together making a logically connected service. Specification. Based on this reasoning we have the following BLE specification:


### Service: Presentation

UUID: **<code>d3560000-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provides the generic interface for devices which can take on the presentation role, that is which show stimuli to the user for which the brain response will be used to make selections later.

Characteristic:  Stimulus  State:

 UUID: **<code>d3560001-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

  Provides the characteristic to communicate the current device stimulus state to the decoder by formatting STIMULUSEVENT messages.

Properties:

Variable length

Read, Write, Notify.

Payload Format:

**Read, Notify**:  encode the current stimulus state in a STIMULUSEVENT message -- Note: this is a *full* message with type, version, length encoding (Later we may remove this technically unnecessary header information), thus it may take *more than* 1 message to communicate a full stimulus state when we have > 8 objectIDs on a single device.  Note however, that as stimulus state’s are ‘latched’ only changed objects need to have their state communicated which may be used to reduce the communication load in resource constrained situations.

(Note: be sure to use the onCharacteristicWrite callback to stream these multi-packet writes in a good way!)

TODO: V2 StimulusEvent spec with binary stimulus state encoding.

Write: encode control messages for the presentation device.  Basically this includes:

   NEWTARGET messages

  SELECTION messages.

Again, we encode the full message spec in the BLE characteristic write (including header, version, length).


### Service: Decoder

UUID: **<code>d3560100-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provides the general interface for the EEG decoder.  Thus, it consumes STIMULUSEVENT messages and generates Prediction messages.

Characteristic:  PredictedTargetProb:

 UUID: **<code>d3560101-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

  Provides the characteristic to communicate the predicted target probability by formatting PREDICTEDTARGETPROB messages.

Properties:

Read, Notify.

Payload Format:

**Read, Notify**:  encode the current predicted target prob as a PREDICTEDTARGETPROB message.  Note this is the full message including headers.

Characteristic:  PredictedTargetDist:

 UUID: **<code>d3560102-b9ff-11ea-b3de-0242ac130004</code></strong>23c

Description:

  Provides the characteristic to communicate the predicted target probability by formatting PREDICTEDTARGETDIST messages.

Properties:

Read, Notify.

Payload Format:

**Read, Notify**:  encode the current predicted target prob as a PREDICTEDTARGETDIST message.  Note: this is a *full* message with type, version, length encoding (Later we may remove this technically unnecessary header information), thus it may take *more than* 1 message to communicate a full prediction state when we have > 8 objectIDs on a single device.  As all messages are both time-stamped and stateless, this will be achieved by simply cutting the messages into smaller pieces to be transmitted one after each other.


### Service: Selection

UUID: **<code>d3560200-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provide the ability to receive and act on selections by the BCI.

Characteristic:  Selection:

 UUID: **<code>d3560201-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

  Provides the characteristic to communicate selections to the output device.

Properties:

Write, Notify


### Service: ScoreOutput

UUID: **<code>d3560300-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provides an output-scoring service, which consumes fitted model parameters and generates a stream of stimulus scores.

Characteristic: **OutputScore**

 UUID: **<code>d3560301-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provides the characteristic to communicate the stimulus output score by sending: OUTPUTSCORE messages.

Properties:

Read, Notify.

Payload Format:

**Read, Notify**:  encode the current output score as an OUTPUTSCORE message.


<table>
  <tr>
   <td><strong>Name: </strong>
<h6>OUTPUTSCORE</h6>


   </td>
   <td><strong>UID: “O”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>OutputScorer</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>ANY component</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
This message sends the current stimulus score information from an acquisition device with local signal processing to BCI server.
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “O”
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
   <td>[1] of uint32
   </td>
   <td>Time this signal quality measure was computed.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>nObjects
   </td>
   <td>[1] of uint8
   </td>
   <td>The number of output scores being sent.
   </td>
  </tr>
  <tr>
   <td>scores
   </td>
   <td>[nout] of float
   </td>
   <td>The actual output scores packed as a sequence of float32
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>


Characteristic:  CurrentModel:

 UUID: **<code>d3560302-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provides a characteristic to communicate the current BCI model to the score-output module.

Properties:

Write

Payload Format:

**Write**:  encode the current model as a CURRENTMODEL message.  Note: this is a *full* message with type, version, length encoding (Later we may remove this technically unnecessary header information), thus it *will* take *more than* 1 message to communicate a full stimulus state when we have > 8 objectIDs on a single device.

Characteristic:  CurrentSOSFILTER:

 UUID:  **<code>d3560303-b9ff-11ea-b3de-0242ac130004</code></strong>

Description:

Provides a characteristic to communicate the current BCI filter to use.

Properties:

Write

PayloadFormat:

Write: Encode the IIR as a Second-Order-Sections filter, which is a matrix of 6 x n-Sections.  Thus, we send it as such a matrix. 


<table>
  <tr>
   <td><strong>Name: </strong>
<h6>SOSIIR</h6>


   </td>
   <td><strong>UID: “I”</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Sender:</strong>
<p>
<strong>Decoder</strong>
   </td>
   <td><strong>Receiver:</strong>
<p>
<strong>ANY component</strong>
   </td>
  </tr>
  <tr>
   <td colspan="2" >Purpose:
<p>
Sends the IIR to use as an EEG pre-filter
   </td>
  </tr>
  <tr>
   <td colspan="2" >Format:

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
   <td>[1] of char = “I”
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
   <td>[1] of uint32
   </td>
   <td>Time this data was sent.  Time is measured <strong>in milliseconds</strong> relative to an arbitrary device dependent real-time clock.
   </td>
  </tr>
  <tr>
   <td>iir
   </td>
   <td>[nout] of float
   </td>
   <td>The actual output scores packed as a sequence of float32
   </td>
  </tr>
</table>


   </td>
  </tr>
</table>
