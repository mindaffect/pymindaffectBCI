package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * the TICKTOC utopia message class, used for clock alignment
 */
public class TickTock implements UtopiaMessage {
    public static final int    MSGID      =(int)'T';
    public static final String MSGNAME    ="TICKTOCK";
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
    public int tock; 
    public TickTock(final int tick, final int tock){
        this.timeStamp=tick;
        this.tock     =tock;
    }
    public TickTock(final int timeStamp){
        this.timeStamp=timeStamp;
        this.tock=-1;
    }
    
    /**
     * deserialize a byte-stream to create an instance of this class 
     * @param buffer  - bytebuffer with the data to generate the object from
     * @param version - version of the message payload to decode 
     */ 
    public static TickTock deserialize(final ByteBuffer buffer, int version)
        throws ClientException {
        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = (int) buffer.getInt();
        int tock=-1;
        if( buffer.remaining() >= 4 ) tock = (int) buffer.getInt();
        return new TickTock(timeStamp,tock);
    }
    public static TickTock deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0); // default to version 0 messages
    }
    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);        
        buf.putInt(timeStamp);
        if( tock>0 ) { buf.putInt(tock); }
    }

    public String toString() {
        String str = "t:" + msgName() + " ts:" + timeStamp;
        str = str + " tock:" + tock;
        return str;
    }
};
