package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * the HEARTBEAT utopia message class
 */
public class Heartbeat implements UtopiaMessage {
    public static final int    MSGID      =(int)'H';
    public static final String MSGNAME    ="HEARTBEAT";
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
    public int getVersion(){
        if( this.statemessage==null ) return 0;
        else return 1;
    }

    /**
     * the state of the system as a string, in Ver.1+ messages.
     */
    public String statemessage; 
    public Heartbeat(final int timeStamp, String statemessage){
        this.timeStamp=timeStamp;
        this.statemessage=statemessage;
    }
    public Heartbeat(final int timeStamp){
        this.timeStamp=timeStamp;
        this.statemessage=null;
    }
    
    /**
     * deserialize a byte-stream to create an instance of this class 
     * @param buffer  - bytebuffer with the data to generate the object from
     * @param version - version of the message payload to decode 
     */ 
    public static Heartbeat deserialize(final ByteBuffer buffer, int version)
        throws ClientException {
        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = (int) buffer.getInt();
        String statemessage=null;
        if( version > 0 ) { // decode the new mode info
            statemessage = UTOPIACHARSET.decode(buffer).toString();
        }
        return new Heartbeat(timeStamp,statemessage);
    }
    public static Heartbeat deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0); // default to version 0 messages
    }
    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);        
        buf.putInt(timeStamp);
        if( statemessage!=null ) { // include the modeinfo
            buf.put(statemessage.getBytes(UTOPIACHARSET));
        }
    }

    public String toString() {
        String str = "t:" + msgName() + " ts:" + timeStamp;
        if( statemessage==null )  {
            str = str + " v:" + "NULL";
        } else{
            str = str + " v:" + statemessage;
        }
        return str;
    }
};
