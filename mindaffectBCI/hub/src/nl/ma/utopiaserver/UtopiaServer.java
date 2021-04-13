package nl.ma.utopiaserver;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import nl.ma.utopiaserver.messages.*;
//import SSDP;

import java.io.File;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.net.InetAddress;
import java.net.InetSocketAddress;
//import java.net.StandardSocketOptions;
import java.net.SocketException;
import java.net.SocketAddress;
import java.net.NetworkInterface;
import java.net.DatagramSocket;
import java.net.MulticastSocket;
import java.net.DatagramPacket;
import java.net.Socket;
import java.net.StandardSocketOptions;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.DatagramChannel;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.Set;
import java.util.List;
import java.util.LinkedList;


/*
 * TODOs:
 */


/**
 * UtopiaServer -- server class which opens a server socket, listens for
 * client requests and processes these requests on demand using non-blocking IO
 *
 * @author jason@mindaffect.nl
 */
public class UtopiaServer implements Runnable {
    //static final String TAG = UtopiaServer.class.getSimpleName();
    public static String TAG = "Server:";
    public static int VERBOSITY = 0;
    public static int LOGLEVEL = 1;
    public final static int DEFAULTPORT = 8400;
    public static String UTOPIA_SSDP_SERVER = "utopia/1.1";
    int port;    // default server port
    Selector sel = null; // hold the list of all currently open clients
    boolean disconnectedOnPurpose = false;

    /**
     * return a string name for this class
     */
    public String getName() {
        return TAG;
    }

    final static int MAXBUFFERSIZE = 1024 * 1024 * 4; // 4Mb buffer per client
    ByteBuffer msgbuffer;
    ByteBuffer tmpbuffer;
    DatagramChannel udpChannel = null; // for recieving/sending UDP messages
    DatagramChannel ssdpChannel = null; // for recieving/sending SSDP messages
    DatagramChannel ssdpChannelv6 = null; // for recieving/sending SSDP messages
    ServerSocketChannel tcpChannel = null; // for recieving TCP messages

    TimeStampClockInterface tsclock;
    public final static int DEFAULTWAITTIMEOUT = 100;
    public final static int HEARTBEATINTERVAL_ms = 250;
    public final static int HEARTBEATINTERVALTCP_ms = 1000;
    public final static int LIVENESSINTERVAL_ms = 5000 + 10;
    int nextHeartbeatTime = 0;
    int nextHeartbeatTimeTCP = 0;
    final static int[] UTOPIAEXCLUDEFORWARDMESSAGES = {TickTock.MSGID};

    public int LOGFLUSHINTERVAL = 15000; // flush every 30s
    public static String DEFAULTLOGFILENAME = "UtopiaMessages.log";
    private File logFile = null;
    private BufferedWriter logWriter = null;
    private int nextLogTime = -1;

    /**
     * list of all decoded incomming messages
     */
    List<ServerUtopiaMessage> inmessageQueue;
    // auto-flush the in-message queue?
    public boolean flushInMessageQueue = true;
    Thread serverThread=null;

    /**
     * Print a usage string to stderr
     */
    public static void usage() {
        System.out.println("java -jar UtopiaServer.jar PORT VERBOSITY LOGFILE");
    }

    /**
     * driver method, starts running a server thread in the current thread, and then recieves and sends utopia messages from clients.
     * Handles arguments.
     *
     * @param args <port>
     */
    public static void main(final String[] args) {
        int port = UtopiaServer.DEFAULTPORT;
        if (args.length == 0) {
            usage();
        }
        if (args.length == 1) {
            try {
                port = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
            }
        }
        if (args.length == 2) {
            try {
                VERBOSITY = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
            }
        }
        String logFileName = DEFAULTLOGFILENAME;
        if (args.length == 3) {
            logFileName = args[2];
        }
        // Now run the server, polling for client messages to process
        UtopiaServer utopiaServer = new UtopiaServer();
        utopiaServer.initialize(port, logFileName);
        utopiaServer.run();
    }

    /**
     * mainloop of the utopia server, waits forever for connections and messages
     * <p>
     * runs a infinite loop, waiting for new messages from utopia
     */
    public boolean running=false;
    public void run() {
        this.initialize();
        running=true;
        System.out.println(TAG + "  Waiting for connections/messages");
        while (running) {
            int nnew = getNewMessages(DEFAULTWAITTIMEOUT);
            if (flushInMessageQueue) {
                synchronized (inmessageQueue) {
                    inmessageQueue.clear();
                }
            }
            Thread.yield();
            if( Thread.currentThread().isInterrupted() ){
                running=false;
                System.exit(0);
            }
        }
        System.exit(0);
    }

    public void interrupt() {
        running=false;
        Thread.currentThread().interrupt();
    }

    public Thread startServerThread() {
        return startServerThread(-1,true);
    }

    public Thread startServerThread(int priority, boolean flushInMessageQueue) {
        // start the utopia server in a background thread
        //if( inmessageQueue!=null ) return null;
        this.flushInMessageQueue = flushInMessageQueue;
        serverThread = new Thread(this, "UtopiaHubServer");
        if (priority > 0) serverThread.setPriority(priority);
        serverThread.start();
        return serverThread;
    }

    public UtopiaServer() {
        if (VERBOSITY > -1) System.out.println(TAG + "cons");
        msgbuffer = ByteBuffer.allocate(MAXBUFFERSIZE);
        ((Buffer) msgbuffer).clear();
        tmpbuffer = ByteBuffer.allocate(MAXBUFFERSIZE);
        ((Buffer) tmpbuffer).clear();
        inmessageQueue = new LinkedList<ServerUtopiaMessage>();
        // set the default time-stamp clock
        settimeStampClock(new TimeStampClockInterface(){
            public long getTimeStamp(){ return System.currentTimeMillis(); }
        });
        nextHeartbeatTime = gettimeStamp();
        nextHeartbeatTimeTCP = gettimeStamp();
    }

    public void initialize() {
        initialize(DEFAULTPORT, DEFAULTLOGFILENAME);
    }

    public void initialize(final int port) {
        initialize(port, DEFAULTLOGFILENAME);
    }

    public void initialize(int port, String logFileName) {
        if (tcpChannel != null) {
            System.out.println("Already initialized!");
            return;
        }
        this.port = port > 0 ? port : DEFAULTPORT;
        // setup the TCP listening socket
        if (VERBOSITY > -1) System.out.println(TAG + "open server ports");
        try {
            // setup the socket we're listening for connections on.
            System.out.println(TAG + " Opening Port:" + this.port);
            InetSocketAddress addr = new InetSocketAddress("0.0.0.0", this.port);
            tcpChannel = ServerSocketChannel.open();
            tcpChannel.socket().bind(addr);
            tcpChannel.socket().setReceiveBufferSize(MAXBUFFERSIZE * 8); // 64k recieve buffer
            tcpChannel.configureBlocking(false);

            // setup our selector and register the main socket on it
            sel = Selector.open();
            tcpChannel.register(sel, SelectionKey.OP_ACCEPT);
        } catch (IOException e) {
            System.out.println(TAG + " Couldn't setup TCP server socket");
            System.out.println(e.getMessage());
            System.exit(1);
        }

        // setup the UDP listening socket
        try {
            // setup the socket we're listening for connections on.
            udpChannel = DatagramChannel.open();
            InetSocketAddress addr = new InetSocketAddress("0.0.0.0", this.port);
            System.out.println(TAG + " Opening UDP Port:" + this.port);
            udpChannel.socket().bind(addr);
            udpChannel.socket().setReceiveBufferSize(MAXBUFFERSIZE * 8); // 64k recieve buffer
            udpChannel.configureBlocking(false);
            // setup our selector and register the main socket on it
            udpChannel.register(sel, SelectionKey.OP_READ);
            System.out.println(TAG + " Utopia Server listening on: UDP:" + this.port);
        } catch (IOException e) {
            System.out.println(TAG + " Couldn't setup UDP server socket");
            System.out.println(e.getMessage());
            System.exit(1);
        }

        // setup the SSDP-UDP listening socket
        // Server to process SSDP requests
        SSDP ssdpserver = new SSDP(UTOPIA_SSDP_SERVER);
        //SSDP.VERB=0;
        try {
            System.out.println(TAG + " Opening SSDP-ipv4 Port:");
            // search for the multicastable NIC
            NetworkInterface ni = SSDP.getMultiCastNIC();
            System.out.println("SSDP NIC: " + ni);
            // setup the socket we're listening for connections on.
            ssdpChannel = DatagramChannel.open(java.net.StandardProtocolFamily.INET);
            //ssdpChannel.setOption(StandardSocketOptions.SO_REUSEADDR,true);
            ssdpChannel.socket().setReuseAddress(true);
            ssdpChannel.socket().bind(new InetSocketAddress(SSDP.SSDP_MULTICAST_PORT));
            try {
                ssdpChannel.setOption(java.net.StandardSocketOptions.IP_MULTICAST_LOOP, true);
            } catch ( NoClassDefFoundError ex ){
                System.out.println("not bound to loopback");
            }
            //ssdpChannel.setOption(java.net.SocketOptions.IP_MULTICAST_IF, ni);
            ssdpChannel.join(InetAddress.getByName(SSDP.SSDP_MULTICAST_ADDRESS), ni);
            ssdpChannel.configureBlocking(false);
            // setup our selector and register the main socket on it
            ssdpChannel.register(sel, SelectionKey.OP_READ, ssdpserver);
            System.out.println(TAG + " Utopia Server listening on: SSDP:" + ssdpChannel.socket().getLocalAddress());
        } catch (IOException e) {
            System.out.println(TAG + " Couldn't setup SSDP server socket -- IO exception ");
            System.out.println(e.getMessage());
        } catch ( NoClassDefFoundError ex ){
            System.out.println(TAG + ex.getMessage());
        } catch ( NoSuchMethodError ex){
            System.out.println(TAG + ex.getMessage());
        }
        // and the ip6 SSDP listening socket
        try {
            System.out.println(TAG + " Opening SSDP-v6 Port:");
            NetworkInterface ni = SSDP.getMultiCastNIC();
            // setup the socket we're listening for connections on.
            ssdpChannelv6 = DatagramChannel.open(java.net.StandardProtocolFamily.INET6);
            ssdpChannelv6.socket().setReuseAddress(true);
            ssdpChannelv6.socket().bind(new InetSocketAddress(SSDP.SSDP_MULTICAST_PORT));
            //ssdpChannelv6.setOption(java.net.SocketOptions.IP_MULTICAST_IF, ni);
            try {
                ssdpChannelv6.setOption(java.net.StandardSocketOptions.IP_MULTICAST_LOOP, true);
            } catch ( NoClassDefFoundError ex ){
                System.out.println("not bound to loopback");
            }
            ssdpChannelv6.join(InetAddress.getByName(SSDP.SSDP_MULTICAST_ADDRESS_IPV6), ni);
            ssdpChannelv6.configureBlocking(false);
            // setup our selector and register the main socket on it
            ssdpChannelv6.register(sel, SelectionKey.OP_READ, ssdpserver);
            System.out.println(TAG + " Utopia Server listening on: SSDP:" + ssdpChannel.socket().getLocalAddress());
        } catch (IOException e) {
            System.out.println(TAG + " Couldn't setup SSDP server socket");
            System.out.println(e.getMessage());

        } catch (java.nio.channels.UnsupportedAddressTypeException e) {
            System.out.println(TAG + " Couldn't setup SSDP server socket");
            System.out.println(e.getMessage());
        } catch ( NoClassDefFoundError ex ){
            System.out.println(TAG + "Argh!!");
        } catch ( NoSuchMethodError ex ){
            System.out.println("multicast not supported");
        }

        // BODGE: add the description.xml thingy....
        if (VERBOSITY > -1) System.out.println("write description.xml");
        if (ssdpChannel != null || ssdpChannelv6 != null) {
            String descriptionxml = ssdpserver.getXMLDescription("utopia/1.1", "MindAffect", "http://www.mindaffect.nl", "Utopia-HUB BCI server", "Utopia-HUB_v1.0", null, null);
            // write it to 'html/' to add to the web-server
            try {
                String descriptionxmlfname = "./description.xml";
                System.out.println(TAG + "writing file: " + descriptionxmlfname);
                FileWriter fr = new FileWriter(descriptionxmlfname);
                fr.write(descriptionxml);
                fr.close();
            } catch (IOException ex) {
                System.out.println("Warning couldnt write the description.xml file");
            } catch ( NoClassDefFoundError ex ){
                System.out.println("Huh? lets not worry");
            }
        }
        System.out.println(TAG + " Ports open, waiting for connections");

        // if wanted, setup the file for logging..
        if (VERBOSITY > -1) System.out.println("setup logging file");
        if (LOGLEVEL > 0) {
            // postfix a unique id, before the final .
            if (logFileName.contains(".")) {
                String datestr = new java.text.SimpleDateFormat("yyMMdd_HHmm").format(new java.util.Date());
                if (logFileName.contains(".")) {
                    int idx = logFileName.lastIndexOf(".");
                    logFileName = logFileName.substring(0, idx) + "_" + datestr + logFileName.substring(idx, logFileName.length());
                } else {
                    logFileName = logFileName + datestr;
                }
            }
            try {
                logFile = new File(logFileName);
                logWriter = new BufferedWriter(new FileWriter(logFile, false));
                System.out.println(TAG + "logging new messages to file: " + logFileName);
            } catch (IOException e) {
                System.err.println(TAG + "Problem opening log-file for writing");
                System.err.println(e);
                logFileName = null;
            }
        }

        String hostIDs = new String();
        // Logg all the IPs for this host
        try {
            java.util.Enumeration<java.net.NetworkInterface> ifcs = java.net.NetworkInterface.getNetworkInterfaces();
            while (ifcs.hasMoreElements()) {
                java.net.NetworkInterface ifc = ifcs.nextElement();
                if (ifc.isUp()) {
                    java.util.Enumeration<java.net.InetAddress> addrs = ifc.getInetAddresses();
                    while (addrs.hasMoreElements()) {
                        java.net.InetAddress tmp = addrs.nextElement();
                        if (tmp instanceof java.net.Inet4Address) {
                            hostIDs = hostIDs + "\n" + tmp + ":" + tcpChannel.socket().getLocalPort();
                        }
                    }
                }
            }
        } catch ( IOException e){
            System.out.println(TAG + " Couldn't get network info");
            System.out.println(e.getMessage());
        }

        // display message with useful configuration information:
        String msg = "Listening on Address : \n" + hostIDs;
        if ( logFile != null ){
            msg = msg + "\n\n Saving to :\n" + logFile.getAbsolutePath();
        }
        showDialog("Utopia Server Status",  msg );
        System.out.println(TAG + " Utopia Server Status: \n" + msg);

    }

    @Override
    protected void finalize() throws IOException {
        if (logWriter != null) { // flush file on exit
            logWriter.flush();
        }
    }

    /**
     * Listen for data from clients, recieve and parse it into the incomming queue, send outgoing messages to clients able to receive, return immeadiately if nothing to do.
     */
    public int getNewMessages() {
        return getNewMessages(0);
    }

    /**
     * Listen for data from clients, recieve and parse it into the incomming queue, send outgoing messages to clients able to receive. Wait at timout_ms if nothing to do.
     */
    public int getNewMessages(int timeout_ms) {
        //System.out.println("Getting new messages");System.out.flush();
        // our canned response for now
        int nnew = 0;
        int nch = 0;
        try {
            // check for any channels with stuff to do
            if (timeout_ms > 0)
                nch = this.sel.select(timeout_ms);
            else
                nch = this.sel.selectNow();
        } catch (IOException e) {
            System.out.println(TAG + " Error in select");
            return 0;
        }
        // get time as close as possible to the event receive time...
        //int currenttimeStamp = gettimeStamp();

        // loop over channels with stuff to do 
        Set keys = this.sel.selectedKeys();
        for (SelectionKey key : this.sel.selectedKeys()) {
            if (key.isAcceptable()) {
                // this means that a new client has hit the port our main
                // socket is listening on, so we need to accept the  connection
                // and add the new client socket to our select pool for reading
                // a command later
                // this will be the ServerSocketChannel we initially registered
                // with the selector in main()
                try {
                    ServerSocketChannel sch = (ServerSocketChannel) key.channel();
                    SocketChannel ch = sch.accept();
                    String clientID = ch.socket().getRemoteSocketAddress().toString();
                    System.out.println("\nServer: New Client Connection : " + clientID + "\n");
                    if (VERBOSITY > 2)
                        showDialog("New Client Connection", "Client Address: " + clientID);
                    // N.B. the ch.socket().getRemoteSocketAddress() gives a unique
                    //      identifier for this client connection
                    ch.configureBlocking(false);
                    ch.socket().setTcpNoDelay(true);
                    ch.socket().setReceiveBufferSize(MAXBUFFERSIZE * 8); // 64k recieve buffer
                    ch.socket().setSendBufferSize(MAXBUFFERSIZE * 8); // 64k send buffer
                    ch.register(this.sel, SelectionKey.OP_READ, new ClientInfo(this, ch, gettimeStamp()));
                } catch (IOException e) {
                    System.out.println(TAG + " \nError in NewConnection");
                    System.out.println(e.getMessage());
                }

            } else if (key.isWritable()) {
                // we are ready to send a response to one of the client sockets
                if (key.channel() instanceof SocketChannel) { // only write queue for sockets
                    SocketChannel ch = (SocketChannel) key.channel();
                    ClientInfo ci = (ClientInfo) (key.attachment());
                    try {
                        synchronized (ci.outmessageQueue) {
                            if (VERBOSITY > 0)
                                System.out.println("Send " + ci.outmessageQueue.size() + " ->" + ci.clientID);
                            for (UtopiaMessage msg : ci.outmessageQueue) {
                                writeMessageTCP(ch, msg);
                            }
                            ci.outmessageQueue.clear();
                        }
                        // re-register for reading *ONLY* again
                        key.interestOps(SelectionKey.OP_READ);
                    } catch (IOException ex) {
                        // TODO [] : add logging info on which channel failed..
                        System.out.println(TAG + "Channel write error:" + ex.getMessage());
                        key.cancel(); // deregister w/o channel close
                        showDialog("WARNING: client closed connection", "Error reading from client: " + ci.clientID);
                    }
                }

            } else if (key.isReadable()) {
                if (VERBOSITY > 2) System.out.println(TAG + " New message on socket!");

                // one of our client sockets has sent a message
                // and we're now ready to read it in
                if (key.channel() instanceof DatagramChannel) {
                    if (key.attachment() instanceof SSDP) { // SSDP multicast channel
                        SSDP ssdp = (SSDP) key.attachment();
                        try {
                            ssdp.processRequest((DatagramChannel) key.channel(), false);
                        } catch (IOException ex) {
                            System.out.println(TAG + "UDP Channel write error:" + ex.getMessage());
                        }

                    } else {
                        try {
                            readDatagramChannel((DatagramChannel) key.channel());//,currenttimeStamp);
                        } catch (IOException ex) {
                            System.out.println(TAG + "UDP Channel read error:" + ex.getMessage());
                        }
                    }

                } else { // client specific socket channel info to process
                    ClientInfo ci = (ClientInfo) (key.attachment());
                    try {
                        readSocketChannel(ci);//,currenttimeStamp);
                    } catch (IOException ex) {
                        System.out.println(TAG + "Read error:" + ex.getMessage());
                        key.cancel(); // deregister w/o channel close
                        showDialog("WARNING: client closed connection", "Error reading from client: " + ci.clientID);
                    }
                }
            }
        }
        // Mark all keys as processed
        keys.clear();

        // Check if we should schedule a heartbeat
        try {
            sendHeartbeatIfTimeout();
        } catch (IOException ex) {
            System.out.println(TAG + ex.getMessage());
            //showDialog("WARNING: Something wrong when sending heartbeats!","");
        }
        return nnew;
    }


    private int readSocketChannel(ClientInfo ci) throws IOException {
        int nread = ci.socketChannel.read(ci.buffer);
        int currenttimeStamp = gettimeStamp();
        if (nread < 0) {
            System.out.println(TAG + " \nError in isReadable : couldnt read any data!");
            ((Buffer) ci.buffer).clear();
            throw new IOException(TAG + "Client closed connection!");
        }
        ((Buffer) ci.buffer).flip();
        if (VERBOSITY > 2)
            System.out.println(currenttimeStamp + "ms " +
                    ci.buffer.remaining() + " to read in channel");

        LinkedList<ServerUtopiaMessage> newMessages = decodeMessages(ci.buffer, ci, currenttimeStamp);
        if (VERBOSITY > 1) {
            System.out.println(TAG + " Got: " + newMessages.size() + " messages <- " + ci.clientID + "(TCP)");
            for (ServerUtopiaMessage msg : newMessages) {
                System.out.println("\t" + msg.clientmsg.toString() + "\n");
            }
        }
        // process the new messages for this client
        processNewMessages(newMessages);

        return newMessages.size();
    }


    private int readDatagramChannel(DatagramChannel ch) throws IOException {
        // TODO: read all waiting messages in one go?
        int npkt = 0;
        int gotpkt = 1; // got a packt in last read
        int t0 = gettimeStamp();
        if (VERBOSITY > 1) System.out.println("\n" + TAG + " UDP-start");
        while (gotpkt > 0) {
            gotpkt = readDatagramPacket(ch);//,gettimeStamp());//currenttimeStamp);
            npkt = npkt + gotpkt;
        }
        if (VERBOSITY > 1)
            System.out.println(TAG + "  UDP-end " + npkt + " packets" + " in " + (gettimeStamp() - t0) + " ms");
        return npkt;
    }

    private int readDatagramPacket(DatagramChannel ch) throws IOException {
        // read packet into a general buff & get sender info
        // N.B. UDP **must** have complete message in 1 packet!
        ((Buffer) tmpbuffer).clear();
        SocketAddress sendAdd = ch.receive(tmpbuffer);
        int currenttimeStamp = gettimeStamp();
        ((Buffer) tmpbuffer).flip();
        if (sendAdd == null) return 0; // stop when no pkts read
        String clientID = sendAdd.toString(); // get where msg came from
        if (VERBOSITY > 2) {
            System.out.println(currenttimeStamp + "ms " +
                    tmpbuffer.remaining() + " to read in channel");
        }
        // get the client info this message came from
        // N.B. Allow match only on IP address
        ClientInfo ci = getClientInfo(clientID, true);
        if (ci == null) {
            System.out.println(TAG + " Huh! got UDP message from unknown client: " + clientID);
        }

        LinkedList<ServerUtopiaMessage> newMessages = decodeMessages(tmpbuffer, ci, currenttimeStamp);
        if (VERBOSITY > 1) {
            for (ServerUtopiaMessage msg : newMessages) {
                System.out.println(TAG + " Got: " + msg.clientmsg.toString() + " messages <- " + (msg.ci == null ? "UNKNOWN" : msg.ci.clientID) + "(UDP)\n");
            }
        }

        // process the new messages as comming from this client
        processNewMessages(newMessages);
        return newMessages.size();
    }

    private ClientInfo getClientInfo(String clientID, boolean iponly) {
        SelectionKey key = getClientKey(clientID, iponly);
        if (key != null)
            return (ClientInfo) (key.attachment());
        return null;
    }

    private SelectionKey getClientKey(String clientID, boolean iponly) {
        SelectionKey key = null;
        String clientIDip = getIPfromClientID(clientID);
        for (SelectionKey keyi : this.sel.keys()) {
            if (keyi.attachment() instanceof ClientInfo) {
                ClientInfo ci = (ClientInfo) (keyi.attachment());
                if (ci != null && clientID.equals(ci.clientID)) { // exact match
                    key = keyi;
                    break; // stop immeadiately with exact match
                }
                if (iponly && ci != null) { // ip-only match
                    if (clientIDip.equals(getIPfromClientID(ci.clientID))) {
                        key = keyi; // keep searching for better exact match
                    }
                }
            }
        }
        return key;
    }

    private String getIPfromClientID(String clientID) {
        int i = clientID.indexOf(":");
        if (i > 0) return clientID.substring(0, i);
        return clientID;
    }

    void processNewMessages(LinkedList<ServerUtopiaMessage> newMessages) {
        // Pass1: update the clock tracking info
        for (ServerUtopiaMessage servermsg : newMessages) {
            ClientInfo ci = servermsg.ci;
            if (ci != null) { // Update the clock alignment from the new message info
                ci.updateClockAlignment(servermsg.recievedservertimeStamp, servermsg);
            }
        }

        // Pass2: add new messages to the incomming message queue
        //        respond to ticktocks
        //        forward messages to other clients
        for (ServerUtopiaMessage servermsg : newMessages) {
            UtopiaMessage msg = servermsg.clientmsg;
            ClientInfo ci = servermsg.ci;
            String clientID = servermsg.ci == null ? "UNKNOWN" : servermsg.ci.clientID;

            if (ci != null) { // if valid client to respond to
                // Special case: TickTock, respond immeadiately on UDP
                if (msg.msgID() == TickTock.MSGID) {
                    respondTickTock((TickTock) msg, ci, servermsg.recievedservertimeStamp);
                    continue;
                } else if (msg.msgID() == Subscribe.MSGID) {
                    ci.updateSubscriptions(((Subscribe) msg).messageIDs);
                    continue; // skip to the next message
                }

                // re-write the server-time-stamps with the updated
                // clock alignment info
                int clienttimeStamp = msg.gettimeStamp();
                int servertimeStamp = ci.getOurTime(clienttimeStamp);
                // set server time-stamp to what we estimate it should be
                servermsg.servertimeStamp = servertimeStamp;
                // record in the inmessageQueue
                synchronized (inmessageQueue) {
                    inmessageQueue.add(servermsg);
                }
            }
            // log with the re-computed server-time-stamp, but BEFORE overwriting the client time-stamp
            logNewMessage(servermsg);

            // Forward new messages to other clients.
            // Check if this message should not be forwarded
            boolean forwardmsg = true;
            for (int fwdmsg : UTOPIAEXCLUDEFORWARDMESSAGES) {
                if (msg.msgID() == fwdmsg) {
                    if (VERBOSITY >= 2) System.out.println(TAG + " Not Forwarding message: " + msg);
                    forwardmsg = false;
                    break;
                }
            }
            if (forwardmsg) {
                // rewrite client-tstamp -> server-tstamp before
                // forwarding
                if (VERBOSITY > 2) {
                    System.out.println(TAG + " Adding server time-stamp: "
                            + servermsg.clientmsg.gettimeStamp() + " (client) -> "
                            + servermsg.servertimeStamp + " (server)");
                }
                msg.settimeStamp(servermsg.servertimeStamp);
                sendMessageOtherClients(msg, clientID);
            }
        }

        if (VERBOSITY > 2) {
            System.out.println(TAG + " New message queue size: " + inmessageQueue.size());
        }
    }

    public void logNewMessages(List<ServerUtopiaMessage> newMessages) {
        for (ServerUtopiaMessage msg : newMessages) {
            logNewMessage(msg);
        }
    }

    public void logNewMessage(ServerUtopiaMessage msg) {
        if (VERBOSITY >= 0) {
            System.out.print((char) msg.clientmsg.msgID());
            System.out.flush();
        }
        if (logWriter != null) {
            // TODO []: move log writer to separate thread to not block message server!
            try {
                //System.out.println("Writing msg to log"+msg.toString());
                logWriter.write(msg.toString());
                logWriter.newLine();
            } catch (IOException e) {
                System.out.println(TAG + "Error writing log file!");
                System.out.println(TAG + "Exception: " + e);
                // set logWriter to null to disable logging from now on...
                logWriter = null;
            }
        }
        // force to flush the file every XXX seconds?
        int ts = gettimeStamp();
        if (logWriter != null && ts > nextLogTime) {
            try {
                logWriter.flush();
            } catch (IOException e) {
                System.out.println(TAG + "Error flushing log file!");
                logWriter = null;
            }
            nextLogTime = gettimeStamp() + LOGFLUSHINTERVAL;
        }
    }

    private void respondTickTock(TickTock msg, ClientInfo ci, int currenttimeStamp) {
        if (msg.tock < 0 && ci != null) {
            TickTock resp = new TickTock(currenttimeStamp, msg.timeStamp);
            try {
                writeMessageUDP(ci.socketChannel.socket(), resp);
            } catch (IOException ex) {
                System.out.println(TAG + " Exception sending tock in TickTock\n" + ex.getMessage());

            }
        }
    }


    private LinkedList<ServerUtopiaMessage> decodeMessages(ByteBuffer buffer, ClientInfo ci, int currenttimeStamp) {
        // read the message header and message ID info
        UtopiaMessage clientmsg;
        LinkedList<ServerUtopiaMessage> newMessages = new LinkedList<ServerUtopiaMessage>();
        boolean shortmessage = false;
        while (buffer.remaining() > 0) {
            try {
                clientmsg = decodeMessage(buffer);
                ServerUtopiaMessage msg = new ServerUtopiaMessage(clientmsg, ci, currenttimeStamp);
                newMessages.add(msg);   // Add to the message queue
            } catch (ClientException ex) {
                // Incomplete message wait for more data
                if (VERBOSITY > 0) System.out.println(ex.getMessage());
                shortmessage = true;
                break;
            }
        }
        if (buffer.remaining() > 0) { // incomplete message
            // move from position->limit back to start of the buffer, then put position at end
            buffer.compact();
        } else { // read it all, clear the buffer
            ((Buffer) buffer).clear();
        }
        if (shortmessage) {
            System.out.println(buffer);
        }
        return newMessages;
    }

    public UtopiaMessage decodeMessage(ByteBuffer buffer) throws ClientException {
        RawMessage message = RawMessage.deserialize(buffer);
        UtopiaMessage clientmsg = message.decodePayload();
        return clientmsg;
    }

    /**
     * get a valid utopia server time-stamp for the current time
     */
    public TimeStampClockInterface getTimeStampClock(){
        return this.tsclock;
    }
    public int gettimeStamp() {
        return (int) tsclock.getTimeStamp();
    }
    public void settimeStampClock(TimeStampClockInterface tsclock){
        // set the time-stamp clock to use, if not the default
        this.tsclock = tsclock;
    }


    /**
     * Remove and return the first message in the message buffer.  null if empty
     */
    public ServerUtopiaMessage popMessage() {

        ServerUtopiaMessage msg = null;
        synchronized (inmessageQueue) {
            if (!inmessageQueue.isEmpty())
                msg = this.inmessageQueue.remove(0);
        }
        return msg;
    }

    /**
     * remove a return all messages waiting in the incomming message buffer, empty array of empty.
     */
    public List<ServerUtopiaMessage> popMessages() {
        LinkedList<ServerUtopiaMessage> newmessages = new LinkedList<ServerUtopiaMessage>();
        synchronized (inmessageQueue) {
            if (!inmessageQueue.isEmpty()) {
                newmessages.addAll(inmessageQueue);
                inmessageQueue.clear();
            }
        }
        return newmessages;
    }

    /**
     * test if the incomming message queue is empty.
     */
    public boolean messageQueueIsEmpty() {
        return this.inmessageQueue.isEmpty();
    }

    /**
     * Add message to outqueue for *all* utopia clients
     * returns if was a client to which we should send the message
     */
    public boolean sendMessageAllClients(UtopiaMessage msg) {
        return sendMessageOtherClients(msg, null);
    }


    /**
     * Add message to outqueue for *all* utopia clients *except* the given one
     * returns if was a client to which we should send the message
     */
    public boolean sendMessageOtherClients(UtopiaMessage msg, String exceptClientID) {
        // re-register all connections as writable
        if (VERBOSITY >= 2)
            System.out.println(TAG + " Sending: " + msg + " to all clients EXCEPT " + exceptClientID);
        boolean iswriteclient = false; // is some client we could send to?
        for (SelectionKey key : this.sel.keys()) {
            if (!key.isValid()) continue; // skip cancelled keys
            try {
                SocketChannel ch = (SocketChannel) key.channel();
                // only if not the excluded client
                String clientID = ch.socket().getRemoteSocketAddress().toString();
                //System.out.println(TAG+" Client: " + clientID + " exceptClient: " + exceptClientID);
                ClientInfo ci = (ClientInfo) key.attachment();
                if (exceptClientID == null || !exceptClientID.equalsIgnoreCase(clientID)) {
                    //System.out.println(TAG+" Sending to this client: " + clientID);
                    // TODO: [] for efficiency, pre-serialize the message?
                    if (ci != null) {
                        // send message now, or queue if write buffer full
                        ci.sendOrQueueMessage(msg);
                        if (!ci.outmessageQueue.isEmpty()) {
                            // messages in the out queue, register to write later
                            key.interestOps(SelectionKey.OP_WRITE | SelectionKey.OP_READ);
                            iswriteclient = true;
                        }

                    }
                }
            } catch (ClassCastException ex) { // skip the server socket / UDP socket
            }
        }
        return iswriteclient;
    }


    /**
     * Immeadiately send a message to all output clients
     */
    public void writeMessageAllClientsUDP(UtopiaMessage msg) throws IOException {
        for (SelectionKey key : this.sel.keys()) {
            if (!key.isValid()) continue; // skip cancelled keys
            if (key.channel() instanceof SocketChannel) {
                SocketChannel ch = (SocketChannel) key.channel();
                writeMessageUDP(ch.socket(), msg);
            }
        }
    }

    /**
     * send heartbeat messages with the given delays between messages.
     *
     * @param delays_ms - array of delays *between* messages.
     * @param onTCP     - flag if use the UDP or TCP port for the messages
     */
    public void sendHeartbeats(int[] delays_ms, boolean onTCP) throws IOException {
        if (onTCP) {
            sendMessageAllClients(new Heartbeat(gettimeStamp()));
        } else {
            writeMessageAllClientsUDP(new Heartbeat(gettimeStamp()));
        }
        if (delays_ms != null) {
            for (int i = 0; i < delays_ms.length; i++) {
                // TODO: don't sleep the main thread!
                try {
                    Thread.sleep(delays_ms[i]);
                } catch (InterruptedException ex) {
                    break;
                }
                if (onTCP) {
                    sendMessageAllClients(new Heartbeat(gettimeStamp()));
                } else {
                    writeMessageAllClientsUDP(new Heartbeat(gettimeStamp()));
                }
            }
        }
    }


    /**
     * send heartbeat messages *on UDP* if inter-heartbeat timeout has expired
     */
    public void sendHeartbeatIfTimeout() throws IOException {
        // Check if we should schedule a heartbeat
        int ts = gettimeStamp();
        //System.out.println("SHBITO: ts="+ts+" nhbt="+nextHeartbeatTimeTCP);
        if (ts > nextHeartbeatTime) {
            if (VERBOSITY > 1)
                System.out.println(TAG + " Heartbeat timeout! ts=" + ts + " hbt=" + nextHeartbeatTime);
            else if (VERBOSITY >= 0)
                System.out.println("h+");
            // N.B. update the timeout first! to avoid recursion
            nextHeartbeatTime = ts + HEARTBEATINTERVAL_ms;
            sendHeartbeats(null, false);
        }
        if (ts > nextHeartbeatTimeTCP) {
            nextHeartbeatTimeTCP = ts + HEARTBEATINTERVALTCP_ms;
            sendHeartbeats(null, true); // 1 heartbeat on TCP
            if (VERBOSITY >= 0) // N.B. hb on new line
                System.out.println("H+");
        }
    }


    /**
     * Check for client liveness, and display warning diaglog if not live
     */
    public boolean checkClientLiveness() {

        int currenttimeStamp = gettimeStamp();
        boolean alllive = true;
        for (SelectionKey key : this.sel.keys()) {
            try {
                SocketChannel ch = (SocketChannel) key.channel();
                ClientInfo ci = (ClientInfo) (key.attachment());
                if (currenttimeStamp - ci.getlastMessageTime() > LIVENESSINTERVAL_ms) {
                    alllive = false;
                    if (ci.warningSent == 0) { // limit the rate at which we warn
                        String msg = "Client: " + ch.socket().getRemoteSocketAddress();
                        System.out.println(TAG + " Warning: Client Not Responding: " + msg);
                        showDialog("WARNING: Client Not Responding", msg);
                        ci.warningSent = currenttimeStamp; // record when we sent the warning
                    }
                }
                //ch.register(this.sel, SelectionKey.OP_WRITE);
            } catch (ClassCastException ex) { // skip the server socket
            }
        }
        return alllive;
    }

    /**
     * Immeadiately send a message on a given socketchannel
     */
    public int writeMessageTCP(SocketChannel ch, UtopiaMessage msg) throws IOException {
        int nwrite = -1;
        synchronized (ch) {
            if (VERBOSITY > 1)
                System.out.println(TAG + " TCP: Sending " + msg.toString() + " -> " + ch.socket().getRemoteSocketAddress());
            nwrite = ch.write(serializeMessage(msg, msgbuffer));
        }
        return nwrite;
    }

    /*
     * Immeadiately send message on UDP port to indicated client
     */
    public void writeMessageUDP(Socket clientSocket, UtopiaMessage msg) throws IOException {
        if (VERBOSITY > 1)
            System.out.println(TAG + " UDP: Sending " + msg.toString() + " -> " + clientSocket.getRemoteSocketAddress());
        synchronized (udpChannel) {
            udpChannel.send(serializeMessage(msg, msgbuffer), clientSocket.getRemoteSocketAddress());
        }
    }

    /**
     * convert a utopia message into a wire-compatiable byte stream
     *
     * @param msg    - the utopia message to send.
     * @param buffer - the byte-buffer to put the message into, N.B. is cleared!
     */
    public ByteBuffer serializeMessage(UtopiaMessage msg, ByteBuffer buffer) {
        // serialize Payload into tempory buffer (to get it's size)
        ((Buffer) tmpbuffer).clear();
        msg.serialize(tmpbuffer);
        ((Buffer) tmpbuffer).flip();
        //System.out.println(TAG+" Message Payload size" + tmpbuffer.remaining());

        // serialize the full message with header into the buffer
        ((Buffer) buffer).clear();
        RawMessage.serialize(buffer, msg.msgID(), msg.getVersion(), tmpbuffer);
        ((Buffer) buffer).flip();
        //System.out.println(TAG+" Message total size" + buffer.remaining()
        return buffer;
    }

    /**
     * safely show a dialog window, fail gracefully if in non-gui situation.
     */
    void showDialog(String title, String msg) {
        try {
            // TODO: [] Tidy this code up?
            javax.swing.JOptionPane p = new javax.swing.JOptionPane(msg);
            javax.swing.JDialog d = p.createDialog(null, title);
            d.setModal(false);
            d.show();
        } catch (java.awt.AWTError ex) {
        } catch (NoClassDefFoundError ex) {
        } catch (java.lang.Error ex) {
        } catch (Exception ex) {
        }
    }

};
