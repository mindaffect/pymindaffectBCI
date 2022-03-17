package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import nl.ma.utopiaserver.ClientException;

/**
 * the MODECHANGE utopia message class, which has a time-stamp and a mode-string.
 */
public class Log implements UtopiaMessage {
    public static final int MSGID         =(int)'L';
    public static final String MSGNAME    ="LOG";

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
    public int getVersion(){  return 0; }
    
    public String logmsg;

    public Log(final int timeStamp, String logmsg){
        this.timeStamp=timeStamp;
        this.logmsg  =logmsg;
    }

    /**
     * deserialize a byte-stream to create an instance of this class 
     */ 
    public static Log deserialize(final ByteBuffer buffer)
        throws ClientException {
        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        // get the new mode string -- N.B. assumed UTF-8 encoded        
        final String logmsg = UTOPIACHARSET.decode(buffer).toString();
        return new Log(timeStamp,logmsg);
    }
    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        // send the string
        buf.put(logmsg.getBytes(UTOPIACHARSET));
    }    
	public String toString() {
		 String str= "t:" + msgName() + " ts:" + timeStamp + " msg:" + logmsg;
		 return str;
	}

    // Field-trip buffer serialization
    public String getType(){  return msgName(); }
    public void getValue(final ByteBuffer buf){ // N.B. only the string is put
        buf.order(UTOPIABYTEORDER);
        buf.put(this.logmsg.getBytes(UTOPIACHARSET));
    }    
};
