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
public class OutputScore implements UtopiaMessage {
    public static final int MSGID = (int) 'O';
    public static final String MSGNAME = "OUTPUTSCORE";

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
     * The stimulus state for the objectIDs given in objIDs
     */
    public float[] scores;

    public OutputScore(final int timeStamp, final int scores) {
        this.timeStamp = timeStamp;
        this.scores = new float[1];
        this.scores[0] = scores;
    }

    public OutputScore(final int timeStamp, final float[] scores) {
        this.timeStamp = timeStamp;
        this.scores = new float[scores.length];
        System.arraycopy(scores, 0, this.scores, 0, scores.length);
    }

    /**
     * deserialize a byte-stream to create an instance of this class
     *
     * @throws exception if the byte-stream does not contains a validly encoded STIMULUSEVENT
     */
    public static OutputScore deserialize(final ByteBuffer buffer, int version)
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
        int size = nObjects;

        // Check if size and the number of bytes in the buffer match
        if (buffer.remaining() < size) { // BODGE: allow for over-long message payloads...
            throw new
                    ClientException("Defined size of data and actual size do not match.");
        }

        // extract into the scores vector
        float[] scores = new float[nObjects];
        for (int i = 0; i < nObjects; i++) {
            scores[i] = buffer.getFloat();
        }
        return new OutputScore(timeStamp, scores);
    }

    public static OutputScore deserialize(final ByteBuffer buffer)
            throws ClientException {
        return deserialize(buffer, 0);
    }

    /**
     * serialize this instance into a byte-stream in accordance with the message spec.
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        buf.put((byte) scores.length);
        for (int i = 0; i < scores.length; i++) {
            buf.putFloat(scores[i]);
        }
    }

    public String toString() {
        String str = "t:" + msgName() + " ts:" + timeStamp;
        str = str + " v[" + scores.length + "]:";
        for (int i = 0; i < scores.length; i++) {
            str = str + scores[i] + ",";
        }
        return str;
    }
};
