package nl.ma.utopiaserver;
import nl.ma.utopiaserver.messages.*;


/**
 * Wrapper for UtopiaMessage to attach additional
 * meta-information about the message.  This includes server timestamp
 * information, and the source client.
 */
public class ServerUtopiaMessage {
    public final UtopiaMessage clientmsg;
    public final ClientInfo ci;
    // client time stamp mapped to server time
    public int servertimeStamp=Integer.MIN_VALUE;
    // time this message was received at the server
    public int recievedservertimeStamp=Integer.MIN_VALUE;
    public ServerUtopiaMessage(UtopiaMessage clientmsg){
        this(clientmsg,null,Integer.MIN_VALUE);
    }
    public ServerUtopiaMessage(UtopiaMessage clientmsg, ClientInfo ci,int recievedservertimeStamp){
        this.clientmsg=clientmsg;
        this.ci       =ci;
        this.recievedservertimeStamp=recievedservertimeStamp;
        this.servertimeStamp=recievedservertimeStamp;
    }
    public String toString(){
        String clientID=ci!=null?ci.clientID:"UNKNOWN";
        return "rts:" + recievedservertimeStamp + " sts:" + servertimeStamp + "  " + clientmsg + " <-" + clientID;
    }
}
