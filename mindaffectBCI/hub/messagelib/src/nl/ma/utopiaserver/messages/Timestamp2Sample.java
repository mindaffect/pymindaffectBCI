package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * the TIMESTAMP2SAMPLE utopia message class, used for clock alignment
 */
public class Timestamp2Sample implements UtopiaMessage {
    public static final int    MSGID      =(int)'Z';
    public static final String MSGNAME    ="TIMESTAMP2SAMPLE";
    /**
     * get the unique message ID for this message type
     */
    public int msgID(){ return MSGID; }
    /**
     * get the unique message name, i.e. human readable name, for this message type
     */
    public String msgName(){ return MSGNAME; }

    public int timeStamp;
    /**
     * get the time-stamp for this message 
     */
    public int gettimeStamp(){return this.timeStamp;}
    /**
     * set the time-stamp information for this message.
     */
    public void settimeStamp(int ts){ this.timeStamp=ts; }
    /**
     * get the version of this message
     */
    public int getVersion(){ return 0; }

    /**
     * the state of the system as a string, in Ver.1+ messages.
     */
    public double ourclock;
    public double theirclock; 
    public Timestamp2Sample(final int ts, final double ourclock, final double theirclock){
        this.timeStamp=ts;
        this.ourclock=ourclock;
        this.theirclock=theirclock;
    }
    
    /**
     * deserialize a byte-stream to create an instance of this class 
     * @param buffer  - bytebuffer with the data to generate the object from
     * @param version - version of the message payload to decode 
     */ 
    public static Timestamp2Sample deserialize(final ByteBuffer buffer, int version)
        throws ClientException {
        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = (int) buffer.getInt();
        double ourclock = (double) buffer.getDouble();
        double theirclock = (double) buffer.getDouble();
        return new Timestamp2Sample(timeStamp,ourclock,theirclock);
    }
    public static Timestamp2Sample deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0); // default to version 0 messages
    }
    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);        
        buf.putInt(timeStamp);
        buf.putDouble(ourclock);
        buf.putDouble(theirclock);
    }

    public String toString() {
        String str = "t:" + msgName() + " ts:" + timeStamp;
        str = str + " oc:" + ourclock + " tc:" + theirclock ;
        return str;
    }
};
