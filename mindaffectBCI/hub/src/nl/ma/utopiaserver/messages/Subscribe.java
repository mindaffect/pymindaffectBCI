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
public class Subscribe implements UtopiaMessage {
    public static final int MSGID         =(int)'B';
    public static final String MSGNAME    ="SUBSCRIBE";
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
    public int[] messageIDs;

    public Subscribe(final int timeStamp, final int messageIDs){
        this.timeStamp=timeStamp;
        this.messageIDs = new int[1];
        this.messageIDs[0]=messageIDs;
    }
    public Subscribe(final int timeStamp, final int[] messageIDs){
        this.timeStamp=timeStamp;
        int nObj=messageIDs.length;
        this.messageIDs   =new int[nObj];
        System.arraycopy(messageIDs,0,this.messageIDs,0,nObj);
    }
    public Subscribe(final int timeStamp, final String messageIDs){
        this.timeStamp=timeStamp;
        int nObj=messageIDs.length();
        this.messageIDs   =new int[nObj];
        for ( int i=0; i<messageIDs.length(); i++ ){
            this.messageIDs[i]=(int)messageIDs.charAt(i);
        }
        //System.out.println("Subscription to:"+messageIDs);
        //System.out.println("Gives"+this);
    }

    /**
     * deserialize a byte-stream to create an instance of this class 
     */ 
    public static Subscribe deserialize(final ByteBuffer buffer, int version)
        throws ClientException {

        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        // Get number of objects.  TODO[] : robustify this
        int nObjects = buffer.remaining();//sizeof(int);
        //final int nObjects = (int) buffer.get();
        //System.out.println("ts:"+timeStamp+" ["+nObjects+"]");        
        int [] messageIDs   = new int[nObjects];        
        for (int i = 0; i < nObjects; i++) {
            messageIDs[i]    = (int) buffer.get();
        }
        return new Subscribe(timeStamp,messageIDs);
    }
    public static Subscribe deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0);
    }

    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        for( int i=0; i<messageIDs.length; i++) {
            buf.put((byte)messageIDs[i]);
        }
    }
    
    public String toString() {
        String str= "t:" + msgName() + " ts:" + timeStamp ;
        str = str + " v[" + messageIDs.length + "]:";
        for ( int i=0; i<messageIDs.length; i++){
            str = str + messageIDs[i] + "(" + ((char)messageIDs[i]) + "),";
        }
        return str;
	}
};
