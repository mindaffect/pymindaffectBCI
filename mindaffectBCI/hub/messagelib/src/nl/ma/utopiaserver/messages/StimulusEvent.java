package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;

import nl.ma.utopiaserver.ClientException;

/**
 * the STIMULUSEVENT utopia message, which gives the stimulus state for a set of currently active stimulus-object identifiers.
 */
public class StimulusEvent implements UtopiaMessage {
    public static final int MSGID = (int) 'E';
    public static final String MSGNAME = "STIMULUSEVENT";

    /**
     * get the unique message ID for this message type
     */
    public int msgID() {
        return MSGID;
    }

    /**
     * get the unique message name, i.e. human readable name, for this message type
     */
    public String msgName() {
        return MSGNAME;
    }

    public int timeStamp;

    /**
     * get the time-stamp for this message
     */
    public int gettimeStamp() {
        return this.timeStamp;
    }

    /**
     * set the time-stamp information for this message.
     */
    public void settimeStamp(int ts) {
        this.timeStamp = ts;
    }

    /**
     * get the version of this message
     */
    public int getVersion() {
        return 0;
    }

    /**
     * The set of unique object identifiers the stimulus state refers to.
     */
    public int[] objIDs;
    /**
     * The stimulus state for the objectIDs given in objIDs
     */
    public int[] objState;

    public StimulusEvent(final int timeStamp, final int objIDs, final int objState) {
        this.timeStamp = timeStamp;
        this.objIDs = new int[1];
        this.objIDs[0] = objIDs;
        this.objState = new int[1];
        this.objState[0] = objState;
    }

    public StimulusEvent(final int timeStamp, final int[] objIDs, final int[] objState) {
        this.timeStamp = timeStamp;
        int nObj = objIDs.length;
        if (objIDs.length != objState.length) {
            System.out.println("objIDs and objState have different lengths");
        }
        if (nObj < objState.length) nObj = objState.length;
        this.objIDs = new int[nObj];
        System.arraycopy(objIDs, 0, this.objIDs, 0, nObj);
        this.objState = new int[nObj];
        System.arraycopy(objState, 0, this.objState, 0, nObj);
    }

    public StimulusEvent(final int timeStamp, final int[] objIDs, final int[] objState, int tgtState) {
        this.timeStamp = timeStamp;
        int nObj = objIDs.length;
        if (objIDs.length != objState.length) {
            System.out.println("objIDs and objState have different lengths");
        }
        if (nObj < objState.length) nObj = objState.length;
        if (tgtState >= 0) // append target info
        {
            this.objIDs = new int[objIDs.length + 1];
            System.arraycopy(objIDs, 0, this.objIDs, 0, nObj);
            this.objIDs[this.objIDs.length - 1] = 0;
            this.objState = new int[objState.length + 1];
            System.arraycopy(objState, 0, this.objState, 0, nObj);
            this.objState[this.objState.length - 1] = tgtState;
        } else {
            this.objIDs = new int[nObj];
            System.arraycopy(objIDs, 0, this.objIDs, 0, nObj);
            this.objState = new int[nObj];
            System.arraycopy(objState, 0, this.objState, 0, nObj);
        }
    }


    /**
     * deserialize a byte-stream to create an instance of this class
     *
     * @throws exception if the byte-stream does not contains a validly encoded STIMULUSEVENT
     */
    public static StimulusEvent deserialize(final ByteBuffer buffer, int version)
            throws ClientException {

        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        // Get number of objects
        final int nObjects = (int) buffer.get();
        //System.out.println("ts:"+timeStamp+" ["+nObjects+"]");

        if (nObjects <= 0) { // BODGE: allow for over-long message payloads...
            throw new ClientException("Illegal number objects <=0");
        }
        int size = nObjects * 2;

        // Check if size and the number of bytes in the buffer match
        if (buffer.remaining() < size) { // BODGE: allow for over-long message payloads...
            throw new
                    ClientException("Defined size of data and actual size do not match.");
        }

        // extract into 2 arrays, 1 for the objIDs and one for the state
        // Transfer bytes from the buffer into a nSamples*nChans*nBytes array;
        int[] objIDs = new int[nObjects];
        int[] objState = new int[nObjects];

        for (int i = 0; i < nObjects; i++) {
            objIDs[i] = (int) buffer.get();
            int state = (int) buffer.get(); // N.B. java loads unsigned as signed!
            objState[i] = state>=0 ? state : state+256;  // signed -> unsigned conversion
        }
        return new StimulusEvent(timeStamp, objIDs, objState);
    }

    public static StimulusEvent deserialize(final ByteBuffer buffer)
            throws ClientException {
        return deserialize(buffer, 0);
    }

    /**
     * serialize this instance into a byte-stream in accordance with the message spec.
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        buf.put((byte) objIDs.length);
        for (int i = 0; i < objIDs.length; i++) {
            buf.put((byte) objIDs[i]);
            buf.put((byte) objState[i]);
        }
    }

    public String toString() {
        String str = "t:" + msgName() + " ts:" + timeStamp;
        str = str + " v[" + objIDs.length + "]:";
        for (int i = 0; i < objIDs.length; i++) {
            str = str + "{" + objIDs[i] + "," + objState[i] + "}";
        }
        return str;
    }
};
