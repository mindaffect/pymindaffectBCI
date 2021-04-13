package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * the RESET utopia class
 */
public class Reset implements UtopiaMessage {
    public static final int MSGID         =(int)'R';
    public static final String MSGNAME    ="RESET";
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
    public int getVersion(){return 0;}

    public Reset(final int timeStamp){
        this.timeStamp=timeStamp;
    }

    /**
     * deserialize a byte-stream to create an instance of this class 
     */ 
    public static Reset deserialize(final ByteBuffer buffer, int version)
        throws ClientException {
        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        return new Reset(timeStamp);
    }
    public static Reset deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0);
    }
    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
    }
    
	public String toString() {
		 String str= "t:" + msgName() + " ts:" + timeStamp;
		 return str;
	}
};
