package nl.ma.utopiaserver;
import nl.ma.utopiaserver.messages.*;
import java.nio.channels.SocketChannel;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.LinkedList;

public class ClientInfo {
    final static String TAG="ClientInfo";
    final static int[] ALWAYSSUBSCRIBEDMESSAGEIDS={Heartbeat.MSGID};
    final static int[] DEFAULTCLIENTSUBSCRIBEDMESSAGES={ Heartbeat.MSGID, ModeChange.MSGID, NewTarget.MSGID, Selection.MSGID, Reset.MSGID, PredictedTargetProb.MSGID, SignalQuality.MSGID  };
    final static int[] clockSyncMessages = { Heartbeat.MSGID, TickTock.MSGID, DataPacket.MSGID  };//StimulusEvent.MSGID
    public String clientID;
    public SocketChannel socketChannel;
    private ClockAlignment client2serverts;
    private int lastMessageTime;
    ByteBuffer buffer;
    public  int warningSent;
    List<UtopiaMessage> outmessageQueue;
    int[] subscribedMessageIDs=null;
    public UtopiaServer U=null;
        
        /**
         * constructor, allocates space for the clients in and out going
         * queues, and for the clock-alignment tracker.
         */
    public ClientInfo(UtopiaServer U, SocketChannel ch){ this(U,ch,Integer.MIN_VALUE); }
    public ClientInfo(UtopiaServer U, SocketChannel ch, int timestamp){
        this.U=U;
        socketChannel = ch;
        if( ch!=null ) {
            clientID = ch.socket().getRemoteSocketAddress().toString();
        } else {
            clientID = "UNKNOWN";
        }
        buffer = ByteBuffer.allocate(U.MAXBUFFERSIZE);((Buffer)buffer).clear();
        // buffer for outgoing messages
        outmessageQueue=new LinkedList<UtopiaMessage>();
        //client2serverts=new ExpWeightedRegression();
        //client2serverts=new MinOffsetTracker();
        // client(=X) may be lower => points can be above the line => lower-bound-line
        client2serverts=new WindowedMinOffsetTracker();//new BoundLineTracker(6*30,true,60*1000,20); //
        lastMessageTime=timestamp;
        warningSent=0;
        subscribedMessageIDs=DEFAULTCLIENTSUBSCRIBEDMESSAGES;
    }

    /**
     * adds new server/client timestamp pair to the clock-alignment tracking system
     *
     * @param ourTime our server timestamp
     * @param theirTime their client timestamp
     */
    private void updateClockAlignment(double ourTime, double theirTime){
        client2serverts.addPoint(theirTime,ourTime); // map from X=theirTime -> Y=ourTime
        //System.out.println(clientID + " addPoint: " + theirTime + "," + ourTime + "   -> " + client2serverts);
        if( U.LOGLEVEL>1 ) 
          U.logNewMessage(new ServerUtopiaMessage(new Log((int)theirTime,"ClockSync"+client2serverts),this,(int)ourTime));
        updatelastMessageTime((int)ourTime);
    }
    public void updateClockAlignment(int currenttimeStamp, ServerUtopiaMessage msg){
        int lastclienttimeStamp=Integer.MIN_VALUE;
        for ( int syncMessage : clockSyncMessages ) { // if is clock sync message
            if ( msg.clientmsg.msgID()==syncMessage ) {
		        lastclienttimeStamp = msg.clientmsg.gettimeStamp();
                break;
            }
        }
        // Using this message to update the clock-alignment
        if( lastclienttimeStamp != Integer.MIN_VALUE ){
            updateClockAlignment(currenttimeStamp,lastclienttimeStamp);
        }
    }
    public void updateClockAlignment(int currenttimeStamp, LinkedList<ServerUtopiaMessage> newMessages){
        // Pass1: get the latest client time-stamp in this set of messages
        int lastclienttimeStamp= Integer.MIN_VALUE;
        for ( ServerUtopiaMessage msg : newMessages ){
            for ( int syncMessage : clockSyncMessages ) { // if is clock sync message
                if ( msg.clientmsg.msgID()==syncMessage ) {
                    lastclienttimeStamp = Math.max(lastclienttimeStamp,msg.clientmsg.gettimeStamp());
                }
            }
        }        
        // Using this message to update the clock-alignment
        if( lastclienttimeStamp != Integer.MIN_VALUE ){
            updateClockAlignment(currenttimeStamp,lastclienttimeStamp);
        }
    }
    
    /**
     * gets estimate of our server timestamp from their client time-stamp
     */
    public int getOurTime(double theirTime){ return (int)(client2serverts.getY(theirTime)); }
    /**
     * gets estimated of their client timestamp from our server timestamp
     */
    public int getThereTime(double ourTime){ return (int)(client2serverts.getX(ourTime)); }
    
    public int getlastMessageTime() { return lastMessageTime; }
    public void updatelastMessageTime(int time) {            
        if( time>lastMessageTime ) lastMessageTime=time; // only if larger
    }
    /**
     * either send immeadiately or queue to send when 
     * channel is ready
     *
     * @param msg the message to be queued
     */
    public void sendOrQueueMessage(UtopiaMessage msg) {
        if( issubscribedMessage(msg) ) {
            if( U.LOGLEVEL>1 )System.out.println("Forwarding:"+msg+"->"+clientID);
            int nwrote=0;
            try {
                nwrote=U.writeMessageTCP(socketChannel, msg);
            } catch ( java.io.IOException ex ) {
                nwrote=-1; // mark as write-failed
            }
            if ( nwrote <= 0 ){ // buffer full, add to queue
                synchronized (outmessageQueue) {
                    outmessageQueue.add(msg);
                }
            }
        } else {
            //System.out.println(TAG+" skipping unsubscribed message: "+msg);
        }
    }
    /**
     * update the list of messages this client is interested in
     */
    public void updateSubscriptions(int []messageIDs){
        subscribedMessageIDs=new int[messageIDs.length];
        System.arraycopy(messageIDs,0,this.subscribedMessageIDs,0,messageIDs.length);
        System.out.print(TAG+"Subscribed "+clientID+" to : [");
        for ( int i=0; i<this.subscribedMessageIDs.length; i++ ) {
            System.out.print((char)this.subscribedMessageIDs[i]);
        }
        System.out.println("]");
    }
    /**
     * check if a message to be sent is one that this client has subscribed to.
     */
    public boolean issubscribedMessage(UtopiaMessage msg) {
        int msgID=msg.msgID();
        if( subscribedMessageIDs==null ) return true;
            // complusory subscriptions
        for ( int submsgID : ALWAYSSUBSCRIBEDMESSAGEIDS ) {
            if ( msgID==submsgID ) return true;
        }
        // choosen subscriptions
        for ( int submsgID : subscribedMessageIDs ){
            if ( msgID==submsgID ) return true;
        }
        if( U.LOGLEVEL>1 )
            System.out.println("Not forwarding unsubscribed message: id=" + msgID + " msg:" + msg + "to client: " + clientID);
        return false;
    }
    public String toString(){
        return clientID + " outmessageQueue " + outmessageQueue.size();
    }
};    
