package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * The DataHeader utopia message class
 */
public class DataHeader implements UtopiaMessage {
    public static final int MSGID         =(int)'A';
    public static final String MSGNAME    ="DATAHEADER";
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

    public float fsample;
    public int nchannels;
    public String[] labels;
    //public String config;
    
    public DataHeader(final int timeStamp, final float fsample,
                      final int nchannels, final String[] labels){
        this.timeStamp=timeStamp;
        this.fsample=fsample;
        this.nchannels=nchannels;
        this.labels=labels;
        //this.config=config;
    }

    /**
     * deserialize a byte-stream to create a NEWTARGET instance
     */ 
    public static DataHeader deserialize(final ByteBuffer buffer, int version)
        throws ClientException {
        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        final int nchannels = buffer.getInt();
        final float fsample = buffer.getFloat();
        String temp = UTOPIACHARSET.decode(buffer).toString();
        String[] labels = temp.split(","); // channel names is comma sep list labels        
        return new DataHeader(timeStamp,fsample,nchannels,labels);
    }
    public static DataHeader deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0);
    }
        /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        buf.putInt(nchannels);
        buf.putFloat(fsample);
        if ( labels!=null ) {
            for ( String lab : labels ) {
                buf.put(lab.getBytes(UTOPIACHARSET));
                buf.put(",".getBytes(UTOPIACHARSET));
            }
        }
    }
    
	public String toString() {
		 return "t:" + msgName() + " ts:" + timeStamp +
           " fs" + fsample +" ch["+nchannels + "]:" +
           String.join(",",labels);
	}    
};
