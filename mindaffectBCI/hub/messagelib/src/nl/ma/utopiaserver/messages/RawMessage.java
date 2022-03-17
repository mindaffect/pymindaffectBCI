package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * Abstract meta class and factory class for generic Utopia Messages
 * converting to and from byte-streams, to RawMessage or UtopiaMessage child classes
 */
public class RawMessage {
   public static int VERBOSITY=0;
   public final int msgID;
	public final int version;
	public final ByteBuffer msgbuffer;
	public final ByteOrder order;

    /**
     * Construct a general utopia message, header + payload.
     * @param msgID - the message ID
     * @param version - version number for the message
     * @param msgbuffer - a byte-buffer for the message payload
     */
    public RawMessage(final int msgID, final int version,
                  final ByteBuffer msgbuffer) {
		this.version = version;
		this.msgID = msgID;
		this.msgbuffer = msgbuffer;
		this.order = UtopiaMessage.UTOPIABYTEORDER;
	}
	public RawMessage(final int msgID, final int version,
                  final ByteBuffer msgbuffer, final ByteOrder order) {
		this.version = version;
		this.msgID = msgID;
		this.msgbuffer = msgbuffer;
		this.order = order;
	}
            
    /**
     * get the unique message ID for this message type
     */
    public int msgID(){ return msgID; }

    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(ByteBuffer outbuffer){
        outbuffer.order(order);
        outbuffer.put((byte)msgID);         // msgID
        outbuffer.put((byte)version);       // ver
        outbuffer.putShort((short)msgbuffer.remaining()); // msg size
        outbuffer.put(msgbuffer);           // payload
    }

    /**
     * directly generate a serialized message from the header information and the message payload byte-stream.
     */
    
    public static void serialize(final ByteBuffer buffer,
                                 int msgID,
                                 int version,
                                 final ByteBuffer msgbuffer){
        ByteOrder order=UtopiaMessage.UTOPIABYTEORDER;
        // BODGE! : for byte-order to native..
        buffer.order(order);
        buffer.put((byte)msgID);
        buffer.put((byte)version);
        buffer.putShort((short)msgbuffer.remaining());
        buffer.put(msgbuffer);
    }
    
    /**
     * deserialize a byte-stream to create an instance of this class 
     */ 
    public static RawMessage deserialize(byte [] data, int len ) throws ClientException {
        return deserialize(ByteBuffer.wrap(data,0,len));
    }
    public static RawMessage deserialize(final ByteBuffer buffer)
        throws ClientException {
        if( buffer.remaining() < 4 ) {
            throw new ClientException("Not enough in receive buffer for Message header");
        }
        ByteOrder order=UtopiaMessage.UTOPIABYTEORDER;
        buffer.order(order);
        int prevpos = ((Buffer)buffer).position(); // record this so return unmodified buffer if failed to decode

        // get the message ID (byte->string)
        final  int msgID     = (int) buffer.get();
        // get the message version (byte->int)
        final  int version   = (int) buffer.get();
        // get the message size (short->int)
        final int size       = (int) buffer.getShort();

        //System.out.println("Message: ID:" + msgID + "(" + version + ")" + size);

        // Check if size and the number of bytes in the buffer are sufficient
        if (buffer.remaining() < size) { // incomplete message, leave buffer to get rest in next call
            ((Buffer)buffer).position(prevpos);
           throw new ClientException("Not a full messages worth of data in the buffer!");
        } else if ( size<1 ) {            
            String errmsg="Malformed BODY size : "+(char)msgID+"("+msgID+")"+ "ver:" + version + " size:"+size;
            /*int curpos = buffer.position();
              buffer.position(prevpos);
              while ( buffer.remaining()>0 ) errmsg += (char)buffer.get();
              buffer.position(curpos);*/
            System.out.println(errmsg);
            throw new ClientException(errmsg);
        }

        // copy the bytes for the rest of the message
        ByteBuffer msgbuffer = ByteBuffer.allocate(size);
        for ( int i=0; i<size; i++ ) { msgbuffer.put(buffer.get()); }
        msgbuffer.order(order);
        ((Buffer)msgbuffer).rewind();
        
        return new RawMessage(msgID,version,msgbuffer,order);
    }
    
    public UtopiaMessage decodePayload() throws ClientException{
        // Decode the payload
        UtopiaMessage evt=null;
        try { 
            if( msgID == StimulusEvent.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + StimulusEvent.MSGNAME + " message");
                evt = StimulusEvent.deserialize(msgbuffer,version);
            
            } else if ( msgID == PredictedTargetProb.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + PredictedTargetProb.MSGNAME + " message");
                evt = PredictedTargetProb.deserialize(msgbuffer,version);

            } else if ( msgID == PredictedTargetDist.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + PredictedTargetDist.MSGNAME + " message");
                evt = PredictedTargetDist.deserialize(msgbuffer,version);

            } else if ( msgID == ModeChange.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + ModeChange.MSGNAME + " message");
                evt = ModeChange.deserialize(msgbuffer,version);
                
            } else if ( msgID == Reset.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Reset.MSGNAME + " message");
                evt = Reset.deserialize(msgbuffer,version);
                
            } else if ( msgID == NewTarget.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + NewTarget.MSGNAME + " message");
                evt = NewTarget.deserialize(msgbuffer,version);
                
            } else if ( msgID == Heartbeat.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Heartbeat.MSGNAME + " message");
                evt = Heartbeat.deserialize(msgbuffer,version);
                
            } else if ( msgID == SignalQuality.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + SignalQuality.MSGNAME + " message");
                evt = SignalQuality.deserialize(msgbuffer,version);
                
            } else if ( msgID == Log.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Log.MSGNAME + " message");
                evt = Log.deserialize(msgbuffer);
                
            } else if ( msgID == Selection.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Selection.MSGNAME + " message");
                evt = Selection.deserialize(msgbuffer);
                
            } else if ( msgID == TickTock.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + TickTock.MSGNAME + " message");
                evt = TickTock.deserialize(msgbuffer,version);
                
            } else if ( msgID == Timestamp2Sample.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Timestamp2Sample.MSGNAME + " message");
                evt = Timestamp2Sample.deserialize(msgbuffer,version);
            
            } else if ( msgID == DataPacket.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + DataPacket.MSGNAME + " message");
                evt = DataPacket.deserialize(msgbuffer,version);
                
            } else if ( msgID == DataHeader.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + DataHeader.MSGNAME + " message");
                evt = DataHeader.deserialize(msgbuffer,version);
                
            } else if ( msgID == Subscribe.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Subscribe.MSGNAME + " message");
                evt = Subscribe.deserialize(msgbuffer,version);

            } else if ( msgID == OutputScore.MSGID ) {
                if ( VERBOSITY>2 )
                    System.out.println("Trying to read " + Subscribe.MSGNAME + " message");
                evt = OutputScore.deserialize(msgbuffer,version);

            } else {
                throw new ClientException("Unsupported Message type: " + msgID);
            }
        } catch ( java.nio.BufferUnderflowException ex ) {
            throw new ClientException("Parse error, payload too short with Message type: " + msgID);
        } catch ( java.lang.NoClassDefFoundError ex ){
            throw new ClientException("Parse error, could not find class def: " + msgID);
        }
        if ( VERBOSITY>1 )
            System.out.println("Got message: " + evt.toString());
        return evt;
    }
    
	@Override
	public String toString() {
       return "{t:"+Integer.toString(this.msgID)+
           "."+Integer.toString(this.version) +
           " [" + Integer.toString(this.msgbuffer.capacity()) + "] }";
	}
}
