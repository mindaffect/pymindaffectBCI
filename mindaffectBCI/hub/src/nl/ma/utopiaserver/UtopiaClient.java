package nl.ma.utopiaserver;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import nl.ma.utopiaserver.messages.*;
import java.io.*;
import java.net.*;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.DatagramChannel;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.SocketChannel;
import java.util.Set;
import java.util.List;
import java.util.LinkedList;
import java.util.Map;

// for auto-discovery of the recogniser
import nl.ma.utopiaserver.SSDP;
// for running our own nested server
import nl.ma.utopiaserver.UtopiaServer;

/**
 * Class providing the message sending/receiving functionality of a utopia Client.
 */
public class UtopiaClient implements Runnable {
    public static String TAG="Client:";
    public static int VERBOSITY=0;
    private static final int DEFAULTREADTIMEOUT=2000;
    private static int HEARTBEATINTERVAL_ms=1000;
    private static int UDPHEARTBEATINTERVAL_ms=500;
    public String host;
    public int port;
    Selector sel = null; // hold the list of all currently open sockets
    DatagramChannel     udpChannel=null; // for UDP messages
    SocketChannel       tcpChannel=null; // for TCP messages
    private static int MAXBUFFERSIZE = UtopiaServer.MAXBUFFERSIZE;
    ByteBuffer inbuffer;
    ByteBuffer outbuffer;
    ByteBuffer tcpbuffer;
    ByteBuffer tmpbuffer;
    ByteBuffer udpbuffer;
    byte[] tmparray;
    TimeStampClockInterface tsclock;
    private int timeout_ms=DEFAULTREADTIMEOUT;
    int nextHeartbeatTime=0;
    int nextUDPHeartbeatTime=0;

    // do we pack multiple datapacket messages in the inqueue into a single one
    public boolean packDataPacketsp=true;//false;//
    
    /**
     * Initialize the buffers and clocks needed for a utopia client connection
     */
    public UtopiaClient(){
        outbuffer = ByteBuffer.allocateDirect(MAXBUFFERSIZE);
        tmpbuffer = ByteBuffer.allocateDirect(MAXBUFFERSIZE);
        tcpbuffer = ByteBuffer.allocateDirect(MAXBUFFERSIZE);
        udpbuffer = ByteBuffer.allocateDirect(MAXBUFFERSIZE);
        inbuffer  = ByteBuffer.allocateDirect(MAXBUFFERSIZE);
        tmparray = new byte[MAXBUFFERSIZE];
        // use the default time-stamp clock
        settimeStampClock(new TimeStampClockInterface(){
            public long getTimeStamp(){ return System.currentTimeMillis(); }
        });
    }

    /**
     * Use auto-discovery to identify the utopia-hub and connect to that
     * hub.
     */
    public String ssdpDiscoverHost(int timeout_ms) throws IOException {
        System.out.println("Searching for utopia servers");
        List<Map<String,String>> utopiaServers=SSDP.discover(UtopiaServer.UTOPIA_SSDP_SERVER,timeout_ms);
        String host="127.0.0.1"; // default to localhost
        int port=UtopiaServer.DEFAULTPORT;
        if( !utopiaServers.isEmpty() ){
            System.out.println(TAG+"Got " + utopiaServers.size() + " utopia servers");
            Map<String,String> utopiaServer=utopiaServers.get(0);
            System.out.println(TAG+"Using server" + utopiaServer);
            if( utopiaServer.containsKey("LOCATION") ) {
                host=utopiaServer.get("LOCATION");
				// strip out the http:// /zxxxx stuff if it's there
                if( host.startsWith("http://") ) host=host.substring("http://".length(),host.length());
				if( host.contains("/") )         host=host.substring(0,host.indexOf('/'));
            } else if ( utopiaServer.containsKey("IP") ) {
                host=utopiaServer.get("IP");
            } 
            // strip host:port if there
            int sep = host.indexOf(':');
            if ( sep>0 ) {
                port=Integer.parseInt(host.substring(sep+1,host.length()));
                host=host.substring(0,sep);
            }
            System.out.println(TAG+"SSDP server: "+host+":"+port);
        } else {
            System.out.println(TAG+"No SSDP responses... falling back on localhost");
        }
        return host;
    }
    
    public boolean connect() throws IOException {
	    return connect(null);
    }

    /**
     * Connect to a given host as a utopia client.
     * @param host - the name of the host to connect to.
     */   
    public boolean connect(String host) throws IOException {
        return connect(host,UtopiaServer.DEFAULTPORT);
    }

    /**
     * Connect to the host:port as a utopia client.
     * @param host - the name of the host to connect to
     * @param port - the port name to connect to.
     */
    public boolean connect(String host, int port) throws IOException {
        if( host==null || host.equals("-") ) {
            host = ssdpDiscoverHost(3000);
        }
        if ( port < 0 ){
            port = UtopiaServer.DEFAULTPORT;
        }
        // setup the TCP socket
        try{
	    if( sel==null ) {
		sel = Selector.open();
	    }
            tcpChannel = SocketChannel.open();
            InetSocketAddress addr = new InetSocketAddress(host ,port);
            System.out.println("Attempting connection to:"+addr);
            // TODO [] : retry the connection
            tcpChannel.connect(addr);
            tcpChannel.configureBlocking(false);
            tcpChannel.socket().setTcpNoDelay(true);
            tcpChannel.socket().setReceiveBufferSize(2*MAXBUFFERSIZE); // 8k recieve buffer
            tcpChannel.register(this.sel, SelectionKey.OP_READ);
        } catch (IOException e) {
	    if ( tcpChannel.isOpen() ) { tcpChannel.close(); }
	    tcpChannel=null;
            System.out.println(TAG+" \nCouldn't setup TCP server socket to:"+host+":"+port);
            System.out.println(e.getMessage());
	    return false;
        }

        // setup the UDP listening socket
        try {
            // setup the socket we're listening for connections on.
            udpChannel = DatagramChannel.open();
            System.out.println(TAG+" Opening UDP Port:" + port);
            udpChannel.socket().setReuseAddress(true);
            try {
                udpChannel.bind(tcpChannel.socket().getLocalSocketAddress());
            } catch ( NoClassDefFoundError ex ){
                System.out.println("Port binding issue");
            } catch ( NoSuchMethodError ex ){
                System.out.println("port binding issue");
            }
            udpChannel.configureBlocking(false);
            udpChannel.socket().setReceiveBufferSize(2*MAXBUFFERSIZE); // 8k recieve buffer
            // setup our selector and register the main socket on it
            udpChannel.register(sel, SelectionKey.OP_READ);
            System.out.println(TAG+" Utopia Server listening on: UDP:" + port);
        } catch (IOException e) {
    	    if ( udpChannel.isOpen() ) { udpChannel.close(); }
	        udpChannel=null;
            System.out.println(TAG+" Couldn't setup UDP channel");
            System.out.println(e.getMessage());
        }      
        this.host=host;
        this.port=port;
        return isConnected(); 
    }

    /**
     * check if we are currently connected to a utopia server
     */
    public boolean isConnected() {
        // TODO [] : improved connection liveness detection..... try a non-blocking read
        if( tcpChannel != null ) return tcpChannel.socket().isConnected();
        else                     return false;
    }

    public String getHostPort(){  return host+":"+port; }

    /**
     * get a current valid client time stamp for the current time.
     */
    public int gettimeStamp(){ return (int)(tsclock.getTimeStamp()); };
    public void settimeStampClock(TimeStampClockInterface tsclock){
        // set the time-stamp clock to use, if not the default
        this.tsclock = tsclock;
        int t=gettimeStamp();
        nextHeartbeatTime=t;
        nextUDPHeartbeatTime=t;
    }
    public TimeStampClockInterface getTimeStampClock(){
        return this.tsclock;
    }

    /**
     * send some initial heartbeat messages with our clock info to initialize the
     * clock alignment between us and the server..
     */
    public void initClockAlign() throws IOException {
        // init align is on TCP?
        // TODO: tcp or udp?
        sendHeartbeats(new int[]{50,50,50,50,50,50,50,50},true); 
    }
    public void sendClockInfo() throws IOException {
        // UDP heartbeat message cascade @ 1ms interval
        sendHeartbeats(new int[]{1,1,1},false); 
    }
    
    public Thread startListenerThread(){
        // return null if listener thread already runnin
        if( inmsgs!=null ) return null;
        Thread listenerThread = new Thread(this,"Listener");
        listenerThread.start();
        return listenerThread;
    }
    
    /**
     * Run a simple utopia client which checks for and prints new incomming messages from the utopia server
     */
    ClockAlignment serverts2ts;
    List<UtopiaMessage> inmsgs=null;
    List<UtopiaMessage> outmsgs=null;
    public void run(){
        inmsgs=new LinkedList<UtopiaMessage>();
        outmsgs = new LinkedList<UtopiaMessage>();
        // server(=X) may be lower => points above the line => lower-bound-line
        serverts2ts=new WindowedMinOffsetTracker();//BoundLineTracker(100,true,120*1000,20);//
        System.out.println(TAG+"Waiting for messages");
        while ( ! Thread.currentThread().isInterrupted() ){

            // try to connect..
            if ( ! isConnected() ){
                // try to connect
                try {
                    connect(host, port);
                } catch ( IOException ex) {

                }
                if ( !isConnected() ){
                    // no point if we're not connected..
                    continue;
                }
            }

            // incomming messages
            try { 
                List<UtopiaMessage> newmsgs=getNewMessagesFromChannel(timeout_ms);
                if( !newmsgs.isEmpty() ) {
                    // TODO [X]: time-stamp tracker and time-stamp rewrite
                    updateClockAlignment(serverts2ts,newmsgs);
                    // add to the incomming message queue & notify clients
                    synchronized ( inmsgs ) {
                        inmsgs.addAll(newmsgs);
                        inmsgs.notifyAll();
                    }
                    if( VERBOSITY > 1 ){
                        for( UtopiaMessage msg : newmsgs ) {
                            System.out.println(TAG+"Got Message: " + msg.toString() + " <- server");
                        }
                    }
                }

                // outgoing messages
                if ( !outmsgs.isEmpty() ){
                    synchronized ( outmsgs ){
                        for ( UtopiaMessage msg : outmsgs ) {
                            sendMessageTCP(msg);
                        }
                        outmsgs.clear();
                        outmsgs.notifyAll();
                    }
                }

            } catch ( IOException ex ) {
                System.out.println(TAG+"Problem reading from stream"); 
                System.exit(-1);
            }
            if( VERBOSITY>0 ) System.out.print('.');System.out.flush();
        }
    }
    public void interrupt(){
	    Thread.currentThread().interrupt();
    }

    protected void updateClockAlignment(ClockAlignment serverts2ts, List<UtopiaMessage> newmsgs){
        int ourts=gettimeStamp();
        int serverts= Integer.MIN_VALUE;
        // Pass 1: get the best updated time-lock and update the alignment
        for( UtopiaMessage msg : newmsgs ) {
            if( msg.msgID()!=Heartbeat.MSGID ) continue; // only HB msgs
            if( msg.gettimeStamp() > serverts ) {
                serverts=msg.gettimeStamp();
            }
        }
        // map from our-time to server-time
        if( serverts>Integer.MIN_VALUE ) {
            serverts2ts.addPoint(serverts,ourts);
            if( VERBOSITY>2 ) {
                System.out.println(TAG+"ts-track updated too: " + serverts2ts);
            }
        }
        // Pass 2: re-write the messages to local time
        for( UtopiaMessage msg : newmsgs ) {
            msg.settimeStamp((int)serverts2ts.getY(msg.gettimeStamp()));
        }
    }
    
    /* 
     * Check if there are messages waiting to be collected
     */
    public boolean hasMessagesAvailable() {
        // TODO [] : make this work
        if( inmsgs!=null ) {  return inmsgs.isEmpty(); }
        return true;
    }

    /**
     * get new messages or return immedaiately which have arrived since the last time we check for messgaes from the utopia server.
     */
    public List<UtopiaMessage> getNewMessages() throws IOException {
        return getNewMessages(0); // non-blocking call
    }
    
    /**
     * wait until time-out for new messages from the utopia server, and return immeadiately when new messages are recieved.
     * @param timeout_ms - timeout for waiting for new messages if none so far.
     */
    public List<UtopiaMessage> getNewMessages(int timeout_ms) throws IOException {

        if( inmsgs==null ) { // get messages directly
            return getNewMessagesFromChannel(timeout_ms);
        }
        
        // we have a listener thread running, just get from it's queue
        synchronized( inmsgs ) { // copy the content to the output
            List<UtopiaMessage> tmpmsgs=new LinkedList<UtopiaMessage>();
            if( inmsgs.isEmpty() && timeout_ms>0 ) {
                try {
                    inmsgs.wait(timeout_ms); // blocking wait
                } catch ( InterruptedException ex ) {
                }
            }
            // pack the datapacket messages if wanted
            if( packDataPacketsp ) {
                tmpmsgs.addAll(packDataPacketMessages(inmsgs));
            } else {
                tmpmsgs.addAll(inmsgs);
            }
            inmsgs.clear();
            return tmpmsgs;
        }
    }
    
    synchronized public List<UtopiaMessage> getNewMessagesFromChannel(int timeout_ms) throws IOException {
        int t0=gettimeStamp();
        int nch=0;
        try {
            // check for any channels with stuff to do
            if( timeout_ms>0 )
                nch = this.sel.select(timeout_ms);
            else
                nch = this.sel.selectNow();
        } catch ( IOException e ) {
            System.out.println(TAG+" Error in select");
            return null;
        }
        if( VERBOSITY > 2 ){
            System.out.println(TAG+" selection time: ts="+ timeout_ms + " / " + (gettimeStamp()-t0));
        }
        
        List<UtopiaMessage> newMessages=new LinkedList<UtopiaMessage>();
        Set keys = this.sel.selectedKeys();
        List<UtopiaMessage> tmpnew=null;
        for ( SelectionKey key : this.sel.selectedKeys() ) {
            if (key.isReadable()) {
                if ( VERBOSITY>2 ) System.out.println(TAG+" New message on socket!");                
                // one of our client sockets has sent a message
                // and we're now ready to read it in
                if( key.channel() instanceof DatagramChannel ) {
                    if ( VERBOSITY>2 ) System.out.println("U");
                    tmpnew=readChannel((DatagramChannel)key.channel());
                } else if ( key.channel() instanceof SocketChannel ) {
                    if ( VERBOSITY>2 ) System.out.println("T");
                    tmpnew=readChannel((SocketChannel)key.channel());
                }
                newMessages.addAll(tmpnew);                
            }
        }
        // Mark all keys as processed
        keys.clear();
        // Check if we should schedule a heartbeat
        try {
            sendHeartbeatIfTimeout();
        } catch ( IOException ex ) {
            System.err.println(TAG + ex.getMessage());
        }
        // if pack-dataPackets - then filter all datapacket messages
        // into a single mega message..
        if ( VERBOSITY>2 ) System.out.println("Got " + newMessages.size() + " new messages");
        if( packDataPacketsp ) {
            newMessages=packDataPacketMessages(newMessages);
        }
        return newMessages;
    }

    synchronized protected List<UtopiaMessage> packDataPacketMessages(List<UtopiaMessage> msgs){
        LinkedList<UtopiaMessage> outmsgs=new LinkedList<UtopiaMessage>();
        LinkedList<float[]> dataPayloads=new LinkedList<float[]>();
        int nmeasurments=0;
        int nsamples=0;
        int lasttimestamp=0;
        for ( UtopiaMessage msg : msgs ) {
            if( msg.msgID()!=DataPacket.MSGID ) {
                outmsgs.add(msg);
                continue;
            }
            DataPacket dp=(DataPacket)msg;
            dataPayloads.add(dp.samples);
            nmeasurments = nmeasurments+dp.samples.length;
            nsamples = nsamples+dp.nsamples;
            lasttimestamp=dp.gettimeStamp();
        }

        // return unchanged msgs if no datapackets to pack
        if ( dataPayloads.size()<=1 ) return msgs;
        if( VERBOSITY>1 )
            System.out.println("Packing "+dataPayloads.size()+" datapakets into 1");
        if ( nmeasurments==0 ) return msgs;

        // copy all datapackets into single mega-message            
        float[] samples=new float[nmeasurments];
        for ( int i=0, offset=0; i<dataPayloads.size(); i++){
            float[] payload=dataPayloads.get(i);
            System.arraycopy(payload,0,samples,offset,payload.length);
            offset += payload.length;
        }
        DataPacket dp = new DataPacket(lasttimestamp,nsamples,nmeasurments/nsamples,samples);
        outmsgs.add(dp);
        return outmsgs;
    }

    private List<UtopiaMessage> readChannel(DatagramChannel ch) throws IOException {
        List<UtopiaMessage> newmsgs=new LinkedList<UtopiaMessage>();
        int npkt=0;
        int gotpkt=1; // got a packt in last read
        int t0=gettimeStamp() ;
        if( VERBOSITY>1 ) System.out.println("\n" + TAG + " UDP-start");
        while ( gotpkt>0 ) {
            List<UtopiaMessage> dp = readDatagramPacket(ch);
            if( dp==null ) break;
            newmsgs.addAll(dp);
            gotpkt=dp.size();
            npkt=npkt+gotpkt;
        }
        if( VERBOSITY>1 ) System.out.println(TAG+"  UDP-end " + npkt + " packets" + " in " + (gettimeStamp()-t0) + " ms"); 
        return newmsgs;
    }

    private List<UtopiaMessage> readDatagramPacket(DatagramChannel ch) throws IOException {
        // read packet into a general buff & get sender info
        // N.B. UDP **must** have complete message in 1 packet!
        ((Buffer)udpbuffer).clear();
        SocketAddress sendAdd = ch.receive(udpbuffer);
        ((Buffer)udpbuffer).flip();
        int t0=gettimeStamp() ;
        if( sendAdd == null ) return null; // stop when no pkts read
        String clientID = sendAdd.toString(); // get where msg came from
        if ( VERBOSITY>2 ){
            System.out.println(t0 + "ms " +
                               udpbuffer.remaining() + " to read in channel");
        }
        List<UtopiaMessage> newMessages = decodeMessages(udpbuffer,t0);
        return newMessages;
    }

    private List<UtopiaMessage> readChannel(SocketChannel ch) throws IOException {
        int nread = ch.read(tcpbuffer);
        if( nread < 0 ) {
            System.out.println(TAG+" \nError in isReadable");
            ((Buffer)tcpbuffer).clear(); 
            throw new IOException(TAG+"Client closed connection!");
        }
        ((Buffer)tcpbuffer).flip();
        int t0=gettimeStamp() ;
        if ( VERBOSITY>2 )
            System.out.println(t0 + "ms " +
                               tcpbuffer.remaining() + " to read in channel");
        List<UtopiaMessage> newMessages = decodeMessages(tcpbuffer,t0);
        return newMessages;
    }


    private List<UtopiaMessage> decodeMessages(ByteBuffer buffer, int currenttimeStamp){
        // read the message header and message ID info
        UtopiaMessage clientmsg;
        LinkedList<UtopiaMessage> newMessages=new LinkedList<UtopiaMessage>();
        int lastclienttimeStamp=-1;
        while ( buffer.remaining() > 0 ){
            try{
                newMessages.add(decodeMessage(buffer));
            } catch ( ClientException ex) {
                // Incomplete message wait for more data
                if( VERBOSITY>1 ) System.out.println(ex.getMessage());
                break;
            }
        }
        if( buffer.hasRemaining() ) { // incomplete message
            // move from position->limit back to start of the buffer, then put position at end
            buffer.compact();
        } else { // read it all, clear the buffer
            ((Buffer)buffer).clear();
        }
        return newMessages;
    }
    
    public UtopiaMessage decodeMessage(ByteBuffer buffer) throws ClientException {
       RawMessage message =RawMessage.deserialize(buffer);
       UtopiaMessage clientmsg = message.decodePayload();
       return clientmsg;
    }



   /**
     * convert a utopia message into a wire-compatiable byte stream
     * @param msg - the utopia message to send.
     * @param buffer - the byte-buffer to put the message into, N.B. is cleared!
     */
    public ByteBuffer serializeMessage(UtopiaMessage msg, ByteBuffer buffer){
        // add time-stamp information if not already there.
        // WARNING: -1 may be a valid time-stamp?
        if( msg.gettimeStamp()==Integer.MIN_VALUE ) msg.settimeStamp(gettimeStamp());
        // serialize Payload into tempory buffer (to get it's size)
        ((Buffer)tmpbuffer).clear();
        msg.serialize(tmpbuffer); 
        ((Buffer)tmpbuffer).flip();
        //System.out.println(TAG+" Message Payload size" + tmpbuffer.remaining());

        // serialize the full message with header into the buffer
        ((Buffer)buffer).clear();
        RawMessage.serialize(buffer,msg.msgID(),msg.getVersion(),tmpbuffer);
        ((Buffer)buffer).flip();
        //System.out.println(TAG+" Message total size" + buffer.remaining()
        return buffer;
    }

    /**
     * send, immeadiately, a given utopia message to the utopia server via TCP connection
     * @param msg - the utopia message to send.
     */
    public void sendMessageTCP(UtopiaMessage msg) throws IOException {
        // Ensure has valid time-stamp, if not set
        if( msg.gettimeStamp()==Integer.MIN_VALUE ) { msg.settimeStamp(gettimeStamp()); }
        // WARNING: -1 may be a valid time-stamp?
        synchronized ( outbuffer ) { // ensure serialize message sending
            ((Buffer)outbuffer).clear();
            tcpChannel.write(serializeMessage(msg,outbuffer));
        }
        sendHeartbeatIfTimeout();
    }
    /**
     * send, immeadiately, a given utopia message to the utopia server via the UDP connection
     * @param msg - the utopia message to send.
     */
    public void sendMessageUDP(UtopiaMessage msg) throws IOException {
        if( msg.gettimeStamp()==Integer.MIN_VALUE ) { msg.settimeStamp(gettimeStamp()); }
        synchronized ( outbuffer ) { // ensure serialize message sending
            ((Buffer)outbuffer).clear();
            udpChannel.send(serializeMessage(msg,outbuffer),tcpChannel.socket().getRemoteSocketAddress());
        }
        sendHeartbeatIfTimeout();
    }    

    /**
     * send, immeadiately, a given utopia message to the utopia server via the TCP connection
     * @param msg - the utopia message to send.
     */
    public void sendMessage(UtopiaMessage msg) throws IOException {
        if ( outmsgs == null ) { // direct mode
            sendMessageTCP(msg);
        } else { // background thread mode
            synchronized (outmsgs){
                outmsgs.add(msg);
            }
        }
    }

    /**
     * send heartbeat messages *on UDP* if inter-heartbeat timeout has expired
     */
    public void sendHeartbeatIfTimeout() throws IOException {
        // Check if we should schedule a heartbeat
        int ts=gettimeStamp();
        if( ts > nextUDPHeartbeatTime ) {
            if( VERBOSITY>1)
                System.out.println(TAG+" Heartbeat timeout! ts="+ ts + " hbt="+nextHeartbeatTime);
            // N.B. update the timeout first! to avoid recursion
            nextUDPHeartbeatTime=ts+UDPHEARTBEATINTERVAL_ms;
            sendHeartbeats(null,false || udpChannel==null ); // heartbeats on UDP, if possible
        }
        if( ts > nextHeartbeatTime ) {
            // N.B. update the timeout first! to avoid recursion
            nextHeartbeatTime=ts+HEARTBEATINTERVAL_ms;
            sendHeartbeats(null,true);
        }
    }
    
     /**
     * send heartbeat messages with the given delays between messages.
     * @param delays_ms - array of delays *between* messages.
     * @param onTCP - flag if use the UDP or TCP port for the messages
     */
    public void sendHeartbeats(int [] delays_ms, boolean onTCP) throws IOException {
        if( onTCP || udpChannel==null ) {
            sendMessageTCP(new Heartbeat(gettimeStamp()));
        } else {
            sendMessageUDP(new Heartbeat(gettimeStamp()));
        }
        if( delays_ms!=null ) {
            for ( int i=0; i<delays_ms.length; i++ ){
                try {
                    Thread.sleep(delays_ms[i]);
                } catch ( InterruptedException ex ) {
                    break;
                }
                if( onTCP || udpChannel==null ) {
                    sendMessageTCP(new Heartbeat(gettimeStamp()));
                } else {
                    sendMessageUDP(new Heartbeat(gettimeStamp()));
                }
            }        
        }
    }

    /**
     * main class for testing the client, connects to the server and sends test messages and listens for incomming messages 
     */
    public static void main(String argv[]) throws Exception {
        String test="log";
        int extraargi=1;
        if( argv.length>0 ) test=argv[0];

        // start our own server thread if wanted
        if( argv.length>1 && ( test.equals("spawnserver")) ) {
            test=argv[1];
            extraargi=2;
                System.out.println("Spawing utopia server");
                UtopiaServer us= new UtopiaServer();
                us.VERBOSITY=-1;
                Thread usthread = new Thread(us);
                usthread.start();
        }

        UtopiaClient utopiaClient = new UtopiaClient();
        boolean isconnected=false;
        while ( !utopiaClient.isConnected() ){
            try { 
                utopiaClient.connect();
            } catch ( IOException ex ) {
                System.out.println(TAG+" Could not connect to server.  waiting");
                ex.printStackTrace();
                Thread.sleep(1000);
            }
        }
        // initialize the time-lock
        utopiaClient.initClockAlign();
        
        if( test.contains("data") ) {
            System.out.println("Running the fake data streamer. Args: fsample, nchannels, blksize");
            float fsample=200;
            if ( argv.length > extraargi) {
                fsample = Float.parseFloat(argv[extraargi]);
                extraargi++;
            }
            int   nchannels=4;
            if (  argv.length > extraargi){
                nchannels = Integer.parseInt(argv[extraargi]);
                extraargi++;
            }
            int   blksize  =(int)(fsample/20f); // stream 20 packets / second
            if ( argv.length > extraargi) {
                blksize = Integer.parseInt(argv[extraargi]);
                extraargi++;
            }
            runFakeDataStreamer(utopiaClient,fsample,nchannels,blksize);
            
        } else if ( test.contains("logall") ){
            // subscribe only the outputmode messages
            System.out.println("Subscribing to only show output relevant messages");
            utopiaClient.sendMessage(new Subscribe(utopiaClient.gettimeStamp(),"MPSNDQE"));
            runMessageLogger(utopiaClient,1000);
        } else if ( test.contains("log") ){
            // subscribe only the outputmode messages
            System.out.println("Subscribing to only show output relevant messages");
            utopiaClient.sendMessage(new Subscribe(utopiaClient.gettimeStamp(),"MPSN"));
            runMessageLogger(utopiaClient,1000);

        } else if ( test.contains("listenlog") ){
            // subscribe only the outputmode messages
            System.out.println("Subscribing to only show output relevant messages");
            utopiaClient.sendMessage(new Subscribe(utopiaClient.gettimeStamp(),"MPSND"));
            Thread listenThread = utopiaClient.startListenerThread();
            runMessageLogger(utopiaClient,1000);

        } else if ( test.contains("logall") ){
            // subscribe only the outputmode messages
            System.out.println("Subscribing to only show output relevant messages");
            utopiaClient.sendMessage(new Subscribe(utopiaClient.gettimeStamp(),"MPSNQEHD"));
            Thread listenThread = utopiaClient.startListenerThread();
            runMessageLogger(utopiaClient,1000);
            
        } else if ( test.contains("selection") ){
            // subscribe only the outputmode messages
            System.out.println("Subscribing to only show output relevant messages");
            utopiaClient.sendMessage(new Subscribe(utopiaClient.gettimeStamp(),"NS"));
            Thread listenThread = utopiaClient.startListenerThread();
            runSelectionInsertion(utopiaClient,1000);
            
        } else { // sending test client
            int offset = utopiaClient.gettimeStamp() % 10;
            try { 
                offset = Integer.parseInt(argv[0]);
            } catch ( NumberFormatException ex ) {                 
            }
            sendTestMessages(utopiaClient,offset);            
        }        
    }

    /**
     * run a simple message logger, showing all received messages.
     *
     * N.B. static method to test how to use the class as a external object
     */
    public static void runMessageLogger(UtopiaClient utopiaClient, int timeout_ms){
        System.out.println(TAG+"Waiting for messages");
        while ( true ){
            try {
                int t0=utopiaClient.gettimeStamp();
                List<UtopiaMessage> newmsgs=utopiaClient.getNewMessages(timeout_ms);
                System.out.println("Got " + newmsgs.size() + " msgs in " + (utopiaClient.gettimeStamp()-t0) + "ms");
                for( UtopiaMessage msg : newmsgs ) {
                    System.out.println(TAG+"Got Message: " + msg.toString() + " <- server");
                }
            } catch ( IOException ex ) {
                System.out.println(TAG+"Problem reading from stream"); 
                System.exit(-1);
            }
            // limit processing rate to test buffer/message queue
            try { Thread.sleep(1000); } catch ( Exception e ) {}
            System.out.print('.');System.out.flush();
        }
    }

    /**
     * run a simple message logger, showing all received messages.
     *
     * N.B. static method to test how to use the class as a external object
     */
    public static void runSelectionInsertion(UtopiaClient utopiaClient, int timeout_ms){
        System.out.println(TAG+"Waiting for messages");
        int selID=-1;
        while ( true ){
            try {
                List<UtopiaMessage> newmsgs=utopiaClient.getNewMessages(timeout_ms);
                System.out.print("\rEnter Selection objectID:");
                if ( System.in.available()>0 ) { // keys to read
                    java.util.Scanner keyboard = new java.util.Scanner(System.in);
                    // read the initial enter.
                    String input = keyboard.nextLine();
                    try {
                        selID=Integer.parseInt(input);
                    } catch ( java.lang.NumberFormatException ex){
                        System.out.println("not a valud object ID");
                    }
                    utopiaClient.sendMessage(new Selection(utopiaClient.gettimeStamp(),selID));
                }
                
            } catch ( IOException ex ) {
                System.out.println(TAG+"Problem reading from stream"); 
                System.exit(-1);
            }
            System.out.print('.');System.out.flush();
        }
    }

    
    /**
     * A fakedata streamer, sending random data
     * with trigger channel support
     */
    public static final int DEFAULTTRIGGERPORT=8300;
    public static Thread startFakeDataThread() throws Exception {
	Thread dataThread = null;
	final UtopiaClient utopiaClient = new UtopiaClient();
	utopiaClient.connect();
	dataThread = new Thread(new Runnable(){
		public void run(){
		    try { 
			runFakeDataStreamer(utopiaClient,200,4,20);
		    } catch ( Exception ex){
			System.out.println(TAG+"Problem in fake data streamer");
		    }
		}
	    });
	dataThread.start();
	return dataThread;
    }
    
    public static void runFakeDataStreamer(UtopiaClient utopiaClient, float fsample, int nchannels, int blocksize) throws Exception{
        // Send the header information
        DataHeader header= new DataHeader(-1,fsample,nchannels,null);
        utopiaClient.sendMessage(header);
        
        // setup the trigger channel        
        DatagramChannel triggerchannel=null;
        try {
            triggerchannel= DatagramChannel.open();
            triggerchannel.socket().bind(new InetSocketAddress(DEFAULTTRIGGERPORT));
            triggerchannel.configureBlocking(false);
        } catch ( Exception e ) {
            triggerchannel=null;
            System.out.println("Something wrong setting up trigger channel");
            System.out.println(e);
        }
        ByteBuffer buf=ByteBuffer.allocate(128);
        float trigamp=0;

        // Do the data streaming
        java.util.Random generator= new java.util.Random();
        DataPacket data = new DataPacket(-1,blocksize,nchannels); // hold the data packet
        int t0=utopiaClient.gettimeStamp();
        double sampInterval_ms = 1000.0/fsample;
        double nextSampTime_ms = t0;
        int nextLogTime_ms    = t0;
        int sleepDuration_ms  = 0;
        int nBlk=0, nSamp=0;
	System.out.println("FakeDataStreamer: "+nchannels+" channels @ " + fsample + "Hz = " + sampInterval_ms + "ms/sample in " + blocksize + " blocks");
        while ( true ) {
            // gen data
            int s=0, c=0, i=0;
            for ( s=0 ; s<data.nsamples; s++){
                // sleep until next sample should be generated
                nextSampTime_ms += sampInterval_ms;
                sleepDuration_ms = (int)(nextSampTime_ms - (double)utopiaClient.gettimeStamp());
		//System.out.println(s+") nst="+nextSampTime_ms+" curt="+utopiaClient.gettimeStamp()+" sd="+sleepDuration_ms);
                if( sleepDuration_ms > 0 ){
                    try { Thread.sleep(sleepDuration_ms); } catch ( InterruptedException e ) {}
                } else if ( sleepDuration_ms < 0 ) {
		    System.out.println("Negative sleep?"+sleepDuration_ms);
		    if( sleepDuration_ms < -100 ) {
			System.out.println("Too far behind, skipping ahead");
			nextSampTime_ms = utopiaClient.gettimeStamp();
		    }
		}
                // generate the data
                for ( c=0; c<data.nchannels-1; c++, i++ ) {
                    data.samples[i] = generator.nextFloat();
                }
                // insert trigger, last channel
                trigamp=0;
                if( triggerchannel!=null ) {
                    ((Buffer)buf).clear();triggerchannel.receive(buf);((Buffer)buf).flip();
                    if(buf.remaining()>0){
                        if ( buf.remaining()<4 ) { // single byte
                            trigamp=(float)buf.get();
                        } else if ( buf.remaining()>=4 ) { // float
                            trigamp=buf.getFloat();
                        }
                        System.out.print("t");
                    }
                }
                data.samples[i]=trigamp;
                i++;
            }
            
            // send the data
            data.settimeStamp(utopiaClient.gettimeStamp()); // ensure valid timestamp
            if( VERBOSITY>2 ) System.out.println(TAG+"Sending:"+data);            
            utopiaClient.sendMessage(data);
            nBlk += 1;
            nSamp+= data.nsamples;

            // logging
            int t=utopiaClient.gettimeStamp();
            if( t > nextLogTime_ms ){
		System.out.print("\n" + nSamp + " " + nBlk + " " + ((t-t0)/1000f) + " " + nSamp*1000/((t-t0)+1) + " (samp,blk,sec,hz)\n");
                System.out.flush();
                nextLogTime_ms = t + 2000;
            }            
        }        
    }

    /**
     * send a set of test messages to the utopia server, using the given server with given offset to uniquely identif the sender.
     */
    public static void sendTestMessages(UtopiaClient utopiaClient,int offset) throws Exception {
                    // write some test messages..
        int[] objIDs = new int[10];
        for ( int i=0; i<objIDs.length; i++ ) objIDs[i]=i+offset;
        int[] objState= new int[objIDs.length];
        
        // MODECHANGE
        {
            ModeChange e = new ModeChange(utopiaClient.gettimeStamp(),"Idle");
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // UDP : HEARTBEAT, v.1.0, time-stamp flurry
        {
            for ( int i=0; i<10; i++ ){
                Heartbeat e = new Heartbeat(utopiaClient.gettimeStamp(),"ClientTest"+i);
                System.out.println(TAG+" Sending: " + e.toString()  + " -> server (UDP)");
                utopiaClient.sendMessageUDP(e);
                Thread.sleep(1);
            }
        }

        // UDP : TICKTOCK
        {
            for ( int i=0; i<10; i++ ){
                TickTock e = new TickTock(utopiaClient.gettimeStamp());
                System.out.println(TAG+" Sending: " + e.toString()  + " -> server (UDP)");
                utopiaClient.sendMessageUDP(e);
                Thread.sleep(1);
            }
            // get the server responses
            List<UtopiaMessage> inmsgs=utopiaClient.getNewMessages(1);
            for( UtopiaMessage msg : inmsgs ) {
                System.out.println(TAG+" Got Message: " + msg.toString() + " <- server");
            }
        }        

        // HEARTBEAT, v.1.0
        {
            Heartbeat e = new Heartbeat(utopiaClient.gettimeStamp(),"ClientTest");
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }


        
            // LOG
        {
            Log e = new Log(utopiaClient.gettimeStamp(),"ClientTest");
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // RESET
        {
            Reset e = new Reset(utopiaClient.gettimeStamp());
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // NEWTARGET
        {
            NewTarget e = new NewTarget(utopiaClient.gettimeStamp());
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // SIGNALQUALITY
        {
            float[] quals={.1f,.2f,.3f,.4f};
            SignalQuality e = new SignalQuality(utopiaClient.gettimeStamp(),quals);
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // SELECTION
        {
            Selection e = new Selection(utopiaClient.gettimeStamp(),10);
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // DATAPACKET
        {
            float[][] dp={{0f,1f,2f,3f},{5f,6f,7f,8f},{9f,10f,11f,12}};
            DataPacket e = new DataPacket(utopiaClient.gettimeStamp(),dp);
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // PREDICTEDTARGETPROB
        for( int i=0; i<10; i++ )
        {
            PredictedTargetProb e = new PredictedTargetProb(utopiaClient.gettimeStamp(),1,((float)(i%10))/10.0f);
            System.out.println(TAG+" Sending: " + e.toString() + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // PREDICTEDTARGETDIST
        for( int i=0; i<10; i++ )
        {
            PredictedTargetDist e = new PredictedTargetDist(utopiaClient.gettimeStamp(),new int[]{i,2,3},new float[]{.9f,.05f,.05f});
            System.out.println(TAG+" Sending: " + e.toString() + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);
        }

        // MODECHANGE
        {
            ModeChange e = new ModeChange(utopiaClient.gettimeStamp(),"Calibrate.Supervised");
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }
        // send some test StimulusEvents
        for ( int i=0; i<5; i++){
            for ( int j=0; j<objState.length; j++) objState[j]=i;
            StimulusEvent e = new StimulusEvent(utopiaClient.gettimeStamp(),objIDs,objState);
            System.out.println(TAG+" Sending: " + e.toString() + " -> server");
            try { 
                utopiaClient.sendMessage(e);
            } catch(IOException ex){ 
                System.out.println(ex);
            }
            Thread.sleep(1000);            
        }

        // MODECHANGE
        {
            ModeChange e = new ModeChange(utopiaClient.gettimeStamp(),"Prediction.static");
            System.out.println(TAG+" Sending: " + e.toString()  + " -> server");
            utopiaClient.sendMessage(e);
            Thread.sleep(1000);            
        }

        // send some more StimulusEvents
        for ( int i=0; i<5; i++){
            for ( int j=0; j<objState.length; j++) objState[j]=i;
            StimulusEvent e = new StimulusEvent(utopiaClient.gettimeStamp(),objIDs,objState);
            System.out.println(TAG+"Sending: " + e.toString() + " -> server");
            try { 
                utopiaClient.sendMessage(e);
            } catch(IOException ex){ 
                System.out.println(ex);
            }
            Thread.sleep(500);
        }

        }
}
