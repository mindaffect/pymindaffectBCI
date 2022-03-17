package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * The SIGNALQUALITY utopia message class, which sends an [0-1] signal quality measure for each connected electrode.
 */
public class SignalQuality implements UtopiaMessage {
    public static final int MSGID         =(int)'Q';
    public static final String MSGNAME    ="SIGNALQUALITY";
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

    /**
     * the array of per-electrode signal qualities.
     */
    public float[] signalQuality;

    public SignalQuality(final int timeStamp, final float signalQuality){
        this.timeStamp=timeStamp;
        this.signalQuality = new float[1];
        this.signalQuality[0]=signalQuality;
    }
    public SignalQuality(final int timeStamp, final float[] signalQuality){
        this.timeStamp=timeStamp;
        int nObj=signalQuality.length;
        this.signalQuality   =new float[nObj];
        System.arraycopy(signalQuality,0,this.signalQuality,0,nObj);
    }

    /**
     * deserialize a byte-stream to create an instance of this class 
     */ 
    public static SignalQuality deserialize(final ByteBuffer buffer, int version)
        throws ClientException {

        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        // Get number of objects.  TODO[] : robustify this
        int nObjects = buffer.remaining()/4;//sizeof(float);
        //final int nObjects = (int) buffer.get();
        //System.out.println("ts:"+timeStamp+" ["+nObjects+"]");        
        float [] signalQuality   = new float[nObjects];        
        for (int i = 0; i < nObjects; i++) {
            signalQuality[i]    = buffer.getFloat();
        }
        return new SignalQuality(timeStamp,signalQuality);
    }
    public static SignalQuality deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0);
    }

    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        for( int i=0; i<signalQuality.length; i++) {
            buf.putFloat(signalQuality[i]);
        }
    }
    
    public String toString() {
        String str= "t:" + msgName() + " ts:" + timeStamp ;
        str = str + " [" + signalQuality.length + "] ";
        for ( int i=0; i<signalQuality.length; i++){
            str = str + signalQuality[i] + ",";
        }
        return str;
	}
};
