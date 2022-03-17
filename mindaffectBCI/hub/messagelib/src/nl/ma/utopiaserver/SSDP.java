package nl.ma.utopiaserver;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.MulticastSocket;
import java.nio.channels.DatagramChannel;
import java.nio.ByteBuffer;
import java.net.SocketTimeoutException;
import java.net.SocketException;
import java.net.InetAddress;
import java.net.SocketAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.UUID;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * SSDP/UPNP server/client implemetation
 */
public class SSDP implements Runnable {
    public final static String TAG="SSDP::";
    public static int VERB=0;
    
    public final static int SSDP_MULTICAST_PORT=1900;
    public final static String SSDP_MULTICAST_ADDRESS="239.255.255.250";
    public final static String SSDP_MULTICAST_ADDRESS_IPV6="FF08::C";
    public final static String NOTIFY_START = "NOTIFY * HTTP/1.1\r\n";
    public final static String SEARCH_START = "M-SEARCH * HTTP/1.1\r\n";
    public final static String HTTP_OK_START = "HTTP/1.1 200 OK\r\n";
    private final static String NOTIFY_TYPE_TEMPLATE = "NTS: %s\r\n";
    private final static String CACHE_CONTROL_TEMPLATE = "CACHE-CONTROL: max-age=%d\r\n";
    private final static String SEARCH_DISCOVER = "MAN: \"ssdp:discover\"\r\n";
    private final static String SYSTEM_IDENTIFIER = System.getProperty("os.name") + "/" + System.getProperty("os.version");
    private final static String NOTIFICATION_TYPE_TEMPLATE = "NT: %s\r\n";
    private final static String UNIQUE_SERVICE_NAME_TEMPLATE = "USN: %s\r\n";
    private final static String SERVER_TEMPLATE = "SERVER: " + SYSTEM_IDENTIFIER + " UPnP/1.1 %s\r\n";
    private final static String LOCATION_TEMPLATE = "LOCATION: %s\r\n";
    private final static String USER_AGENT_TEMPLATE = "USER-AGENT: " + SYSTEM_IDENTIFIER + " UPnP/1.1 %s\r\n";
    private final static String MULTICAST_HOST = "HOST: " + SSDP_MULTICAST_ADDRESS + ":" + SSDP_MULTICAST_PORT + "\r\n";
    private final static String SEARCH_TARGET_TEMPLATE = "ST: %s\r\n";
    private final static String NOTIFY_ALIVE = "ssdp:alive";
    private final static String NOTIFY_UPDATE = "ssdp:update";
    private final static String NOTIFY_BYEBYE = "ssdp:byebyte";
    private final static String SEARCH_ALL = "ssdp:all";
    // general catch all SearchTarget service type queries that all servers should respond to
    public static String [] SEARCH_RESPONSE_IDS = {
        "ssdp:all", "upnp:rootdevice", "ssdp:discover" 
    };
    public final static int DEFAULT_EXPIRE_SECONDS=3600; // 1hr
    
    private final static String NOTIFY_TEMPLATE =
        NOTIFY_START +
        MULTICAST_HOST +
        NOTIFY_TYPE_TEMPLATE +
        NOTIFICATION_TYPE_TEMPLATE +
        UNIQUE_SERVICE_NAME_TEMPLATE;    
    private static  String buildNotifyMessage(String notificationMessageType, String notificationType, String uniqueServiceName) {
        return String.format(NOTIFY_TEMPLATE, notificationMessageType, notificationType, uniqueServiceName);
    }        

    private final static String NOTIFY_ALIVE_TEMPLATE =
        LOCATION_TEMPLATE +
        CACHE_CONTROL_TEMPLATE +
        SERVER_TEMPLATE +
        "\r\n";
    public static  String buildNotifyAliveMessage(String notificationType, String uniqueServiceName, String location, long expireSeconds, String serverSuffix) {
        return buildNotifyMessage(NOTIFY_ALIVE, notificationType, uniqueServiceName) + String.format(NOTIFY_ALIVE_TEMPLATE, location, expireSeconds, serverSuffix) + "\r\n";
    }    
    public static  String buildNotifyUpdateMessage(String notificationType, String uniqueServiceName, String location, long expireSeconds, String serverSuffix) {
        return buildNotifyMessage(NOTIFY_UPDATE, notificationType, uniqueServiceName) + "\r\n";
    }    
    public static  String buildNotifyByeByeMessage(String notificationType, String uniqueServiceName, String location, long expireSeconds, String serverSuffix) {
        return buildNotifyMessage(NOTIFY_BYEBYE, notificationType, uniqueServiceName) + "\r\n";
    }        

    private final static String SEARCH_TEMPLATE =
        SEARCH_START +
        MULTICAST_HOST +
        SEARCH_DISCOVER +
        USER_AGENT_TEMPLATE +
        SEARCH_TARGET_TEMPLATE +
        "MX: %d\r\n" +
        "\r\n";
    public static  String buildSearchMessage(String searchType, int seconds) {
        if( searchType == null ) searchType=SEARCH_ALL;
        return String.format(SEARCH_TEMPLATE, "utopia/1.0", searchType, seconds);
    }


    private final static String SEARCH_RESPONSE_TEMPLATE =
        HTTP_OK_START +
        UNIQUE_SERVICE_NAME_TEMPLATE +
        SEARCH_TARGET_TEMPLATE +
        LOCATION_TEMPLATE +
        CACHE_CONTROL_TEMPLATE +
        SERVER_TEMPLATE +
        "EXT:\r\n" +
        "\r\n";
    public static  String buildSearchResponseMessage(String uniqueServiceName, String searchTarget, String location, long expireSeconds, String serverSuffix) {
        return String.format(SEARCH_RESPONSE_TEMPLATE, uniqueServiceName, searchTarget, location,  expireSeconds, serverSuffix);
    }

    private final static String DESCRIPTION_XML_TEMPLATE =
        "<?xml version=\"1.0\"?>\n" +
        "<root xmlns=\"urn:schemas-upnp-org:device-1-0\">\n"+
        "  <specVersion>\n"+
        "    <major>1</major>\n"+
        "    <minor>0</minor>\n"+
        "  </specVersion>\n"+
        "  <URLBase>%s</URLBase>\n"+
        "  <device>\n"+
        "    <deviceType>urn:schemas-upnp-org:device:Basic:1</deviceType>\n"+
        "    <friendlyName>%s</friendlyName>\n"+
        "    <manufacturer>%s</manufacturer>\n"+
        "    <manufacturerURL>%s</manufacturerURL>\n"+
        "    <modelDescription>%s</modelDescription>\n"+
        "    <modelName>%s</modelName>\n"+
        "    <UDN>uuid:%s</UDN>\n"+
        "    <serviceList>\n%s</serviceList>\n"+
        "    <presentationURL>%s</presentationURL>\n"+
        "  </device>\n"+
        "</root>\n";        
    
    public static String buildXMLDescription(String urlbase,
						String friendlyName,
                                             String manufacture,
                                             String manufactureURL,
                                             String modelDescription,
                                             String modelName,
                                             String uuid,
                                             String serviceList,
                                             String presentationURL){
        return String.format(DESCRIPTION_XML_TEMPLATE, urlbase, friendlyName,manufacture,manufactureURL,modelDescription,modelName,uuid,serviceList,presentationURL);        
    }
    
    public static  Map<String,String> parseMessage(DatagramPacket packet) throws IOException {
        Map<String,String> headers=parseMessage(new String(packet.getData(),"UTF-8"));
        // now add the general information from the packet itself
        headers.put("IP",packet.getAddress().getHostAddress());
        return headers;        
    }
    
    public static  Map<String,String> parseMessage(String packet) {
        HashMap<String, String> headers = new HashMap<String, String>();
        Pattern pattern = Pattern.compile("(.*): (.*)");        
        String[] lines = packet.split("\r\n");        
        // message starts with COMMAND * HTTP/1.1
        String[] bits = lines[0].trim().split("\\s+");
        headers.put("TYPE",bits[0]); 
        for (String line : lines) {
            Matcher matcher = pattern.matcher(line);
            if(matcher.matches()) {
                headers.put(matcher.group(1).toUpperCase(), matcher.group(2));
            }
        }        
        return headers;
    }

    /**
     * Discover any UPNP device using SSDP (Simple Service Discovery Protocol).
     * @param timeout in milliseconds
     * @param searchTarget if null it use "ssdp:all"
     * @return List of parsed matching response messages
     * @throws IOException
     * @see <a href="https://en.wikipedia.org/wiki/Simple_Service_Discovery_Protocol">SSDP Wikipedia Page</a>
     */
    public static List<Map<String,String>> discover(String searchTarget, int timeout, int retries, boolean returnfirst, boolean ipv6) throws IOException {
        ArrayList<Map<String,String>> devices = new ArrayList<Map<String,String>>();
        byte[] sendData;
        byte[] receiveData = new byte[1024];

        String address=SSDP.SSDP_MULTICAST_ADDRESS;
        if ( ipv6 ) address=SSDP.SSDP_MULTICAST_ADDRESS_IPV6;

        /* setup the network socket for send/receive */
        //NetworkInterface ni = getMultiCastNIC();
        List<NetworkInterface> nics = getMultiCastNICs();
        InetAddress inetaddress = InetAddress.getByName(address);
        int port = SSDP_MULTICAST_PORT;
        // make a socket per interface
        List<MulticastSocket> clientSockets = new ArrayList<MulticastSocket>();
        for ( NetworkInterface nic : nics ) {
            MulticastSocket mcs = new MulticastSocket();
            try {
                mcs.setNetworkInterface(nic);
                clientSockets.add(mcs);
            } catch ( IOException ex){

            }
        }

        /* Create the search request */
        String msearch=SSDP.buildSearchMessage(searchTarget,timeout);
        sendData = msearch.toString().getBytes();
        DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, inetaddress, port);

        // Try sending the message and waiting for response multiple times
        for ( int i=0; i<retries; i++) {
            /* Send the request */
            if( VERB>0 ) System.out.println(TAG+"Sending the discovery message to:" + address);
            if( VERB>0 ) System.out.println(TAG+msearch);
            // send query on all sockets
            for ( MulticastSocket clientSocket : clientSockets ) {
                try {
                    clientSocket.send(sendPacket);
                } catch ( IOException ex){

                }
            }

            /* Receive all responses */
            /* TODO[]: use a selector to wait on all interfaces at the same time? */
            if (VERB > 0) System.out.println(TAG + "Waiting for responses");
            // only the 1st socket has the waiting timeout set
            for ( MulticastSocket clientSocket : clientSockets ) {
                clientSocket.setSoTimeout(1);
            }
            clientSockets.get(0).setSoTimeout(timeout);

            // listen on all sockets
            for ( MulticastSocket clientSocket : clientSockets ) {
                while (true) {
                    try {
                        DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);
                        clientSocket.receive(receivePacket);
                        if (VERB >= 0)
                            System.out.println(TAG + "Got response on "+ clientSocket.getNetworkInterface().toString() + "\n" + new String(receivePacket.getData()));
                        Map<String, String> resp = SSDP.parseMessage(receivePacket);
                        if (searchTarget == null ||
                                resp.get("ST").contains(searchTarget)) {
                            if (VERB > 1) System.out.println(TAG + "Added to matching responses");
                            devices.add(resp);
                            if (returnfirst) break; // exit with first match
                        }
                    } catch (SocketTimeoutException e) {
                        break;
                    }
                }
            }
            if( !devices.isEmpty() ) break;
        }
        for ( MulticastSocket clientSocket : clientSockets) {
            clientSocket.close();
        }
        return devices;
    }
    // defaluted versions
    public static List<Map<String,String>> discover() throws IOException {
        return discover(null,3000,5,true,false);
    }
    public static List<Map<String,String>> discover(String searchTarget) throws IOException {
        return discover(searchTarget,3000,5,true,false);
    }
    public static List<Map<String,String>> discover(String searchTarget, int timeout) throws IOException {
        return discover(searchTarget,timeout,5,true,false);
    }

    public static List<Map<String,String>> discoverv6() throws IOException {
        return discover(null,3000,5,true,true);
    }
    public static List<Map<String,String>> discoverv6(String searchTarget, int timeout) throws IOException {
        return discover(searchTarget,timeout,5,true,true);
    }

    @Override
    public  void run() {
        runServer();
    }

    public void runServer() { runServer(false); }
    public void runServer(boolean ipv6) {
        System.out.println(TAG+"SSDP server running for:");
        System.out.println(TAG+"SERVER: "+servertype);
        System.out.println(TAG+"USN: "+usn);
        System.out.println(TAG+"LOCATION:"+location);
        
        String address=SSDP.SSDP_MULTICAST_ADDRESS;
        if( ipv6 ) {
            address=SSDP.SSDP_MULTICAST_ADDRESS_IPV6;
        }
        System.out.println(TAG+"ADDRESS:"+address);
        int port=SSDP.SSDP_MULTICAST_PORT;
        System.out.println(TAG+"PORT:"+port);
        // run as a SSDP service server        
        MulticastSocket socket;
        try {
            InetAddress inetaddress=InetAddress.getByName(address); 
            socket = new MulticastSocket(port);
            socket.setLoopbackMode(false);
            socket.setReuseAddress(true);
            try {
                socket.joinGroup(inetaddress);
            } catch ( SocketException ex ) { // couldn't find device to bind to... set NIC
                NetworkInterface ni = getMultiCastNIC();
                socket.setNetworkInterface(ni);
                socket.joinGroup(inetaddress);
            }
        } catch (Exception e) {
		    System.out.println("Exception making server socket: " + e);
            return;
        }

            byte[] buffer = new byte[8192];
            DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
            while (true ) { 
                try {
                    socket.receive(packet);
                    if( VERB>0 ) System.out.println(TAG+"Got query packet: "+ packet.getAddress() +"\n"+new String(packet.getData()).trim()+"\n");
                    processRequest(packet);
                    // create a new packet and buffer for the next listen
                    buffer = new byte[8192];
                    packet = new DatagramPacket(buffer, buffer.length);
                } catch (SocketTimeoutException e) {
		    System.out.print(".");
                    continue;
                } catch( IOException e ) {
                    System.out.println("Problem parsing query message");
		    System.out.println(e);
                } 
            }
    }
    
    public void processRequest(String requeststr, InetSocketAddress sendAdd) throws IOException, SocketException {
        Map<String,String> request = SSDP.parseMessage(requeststr);
        if ( issearchMatch(request) ) {
            if( VERB>-1 )  System.out.println(TAG+"Query matched.  Sending response to" + sendAdd.toString());
            /* Create the response message */
            String mresp=SSDP.buildSearchResponseMessage(usn,request.get("ST"),location,SSDP.DEFAULT_EXPIRE_SECONDS,servertype);
            if( VERB>1 ) System.out.println(TAG+mresp);
            
            /* Send the response message */
            byte[] sendData = mresp.getBytes();
            DatagramPacket packet = new DatagramPacket(sendData,sendData.length,sendAdd.getAddress(),sendAdd.getPort());
            packet.setData(sendData); // re-use the request packet to get address etc.
            try{
                DatagramSocket clientSocket = new DatagramSocket();
                clientSocket.send(packet);
            } catch ( SocketException ex ) {
            }
        } else {
            if( VERB>1 ) System.out.println(TAG+"Query did *not* match");
        }
    }
    public void processRequest(DatagramChannel ch) throws IOException, SocketException {
        processRequest(ch,false);
    }
    public void processRequest(DatagramChannel ch, boolean printreq) throws IOException, SocketException {
        ByteBuffer buffer= ByteBuffer.allocate(8192);
        SocketAddress sendAdd=ch.receive(buffer);
        String req = new String(buffer.array(),"UTF-8");
        if (printreq) System.out.println("SSDP Query on :" + ch.toString() + "\n" + req.trim() + "\n");
        processRequest(req,(InetSocketAddress)sendAdd);
    }
    public void processRequest(DatagramPacket packet) throws IOException, SocketException {
        processRequest(new String(packet.getData()),(InetSocketAddress)packet.getSocketAddress());
    }

    private boolean issearchMatch(Map<String,String> request){
		String searchTerm = request.get("ST");
      if( searchTerm==null ) return false;
      if( searchTerm.contains(servertype) ) {
          return true; 
      }        
      for ( String s : SEARCH_RESPONSE_IDS ) {
          if( searchTerm.contains(s) ) {
              return true;
          }
      }
      return false;
    }
    
    String servertype=null;
    String usn=null;
    String location=null;
    public SSDP(String servertype, String location, String usn){
        this.servertype=servertype;
        this.location=location;
        this.usn=usn;
    }
    public SSDP(String servertype, String location){
        this.servertype=servertype;
        this.location=location;
        this.usn=guessUSN();        
    }
    public SSDP(String servertype){
        this.servertype=servertype;
        this.location= "http://"+guessLocation()+"/description.xml";        
        this.usn=guessUSN();
    }
    public static String guessUSN() { return UUID.randomUUID().toString(); }
    public static String guessLocation(){ 
        try {
            return guessPrimaryNetworkAddress().getHostAddress();
        } catch ( SocketException ex ) {
            return "127.0.0.1";
        }
    } 
	
    public static InetAddress guessPrimaryNetworkAddress() throws SocketException {
        // assume location is our current address
        // Use this UDP connection trick to get the FQDN for this host
        // see:  https://stackoverflow.com/questions/9481865/getting-the-ip-address-of-the-current-machine-using-java
		InetAddress ip=null;
        for ( java.util.Enumeration<java.net.NetworkInterface> ifcs = java.net.NetworkInterface.getNetworkInterfaces();ifcs.hasMoreElements(); ) {
            java.net.NetworkInterface ifc = ifcs.nextElement();
            if( ifc.isLoopback() || !ifc.isUp() || ifc.isPointToPoint() ) continue; // skip loopback
            for ( java.util.Enumeration<InetAddress> enumipadd=ifc.getInetAddresses(); enumipadd.hasMoreElements(); ){
                InetAddress nicip = enumipadd.nextElement();
                System.out.println(ifc.getDisplayName() + "->" + nicip + " ll=" + nicip.isLinkLocalAddress());
                if( nicip instanceof java.net.Inet4Address ) {
					if ( ip==null ) ip=nicip;
					if ( ip.isLinkLocalAddress() ) ip=nicip; // replace link local
                }
            }
        }
		if ( ip==null ) ip = InetAddress.getLoopbackAddress();
        return ip;
    }

    public static List<NetworkInterface> getMultiCastNICs() throws SocketException {
        // assume location is our current address
        // Use this UDP connection trick to get the FQDN for this host
        // see:  https://stackoverflow.com/questions/9481865/getting-the-ip-address-of-the-current-machine-using-java
        NetworkInterface ni=null;
        InetAddress ip=null;
        int niscore=0;
        List<NetworkInterface> nics= new ArrayList<NetworkInterface>();
        for ( java.util.Enumeration<java.net.NetworkInterface> ifcs = java.net.NetworkInterface.getNetworkInterfaces();ifcs.hasMoreElements(); ) {
            java.net.NetworkInterface ifc = ifcs.nextElement();
            if( !ifc.isUp() || ifc.isPointToPoint() ) continue; // skip loopback
            nics.add(ifc);
        }
        return nics;
    }

    public static NetworkInterface getMultiCastNIC() throws SocketException {
        // assume location is our current address
        // Use this UDP connection trick to get the FQDN for this host
        // see:  https://stackoverflow.com/questions/9481865/getting-the-ip-address-of-the-current-machine-using-java
		NetworkInterface ni=null;
		InetAddress ip=null;
		int niscore=0;
        for ( java.util.Enumeration<java.net.NetworkInterface> ifcs = java.net.NetworkInterface.getNetworkInterfaces();ifcs.hasMoreElements(); ) {
            java.net.NetworkInterface ifc = ifcs.nextElement();
			if( ni==null ) {
			    ni=ifc;
			    niscore=1;
            }
            if( ifc.isLoopback() || !ifc.isUp() || ifc.isPointToPoint() ) continue; // skip loopback
			// search for nic with non-link-local addresses
            for ( java.util.Enumeration<InetAddress> enumipadd=ifc.getInetAddresses(); enumipadd.hasMoreElements(); ){
                InetAddress nicip = enumipadd.nextElement();
				System.out.println(ifc.getDisplayName() + "->" + nicip + " ll=" + nicip.isLinkLocalAddress() + "sl=" + nicip.isSiteLocalAddress() );
				if( niscore<2 && !nicip.isLinkLocalAddress() ) {
				    ni = ifc;
				    niscore = 2;
				} // override link-local
                // override if site-local
                if ( niscore < 3 && nicip.isSiteLocalAddress() ){
                    ni = ifc;
                    niscore = 3;
                }
			}
		}
		System.out.println("NIC="+ni);
        return ni;
    }

    public String getXMLDescription(String friendlyName,
                                    String manufacture,
                                    String manufactureURL,
                                    String modelDescription,
                                    String modelName,
                                    String serviceList,
                                    String presentationURL){
        String baseURL=this.location;
        if( friendlyName==null ) friendlyName="friendlyName";
        if( manufacture==null )  manufacture="MindAffect";
        if( manufactureURL==null )  manufactureURL="http://mindaffect.nl";
        if( modelName==null )    modelName="modelName";
	String uuid = this.usn;
        if( serviceList==null)    serviceList="/";
        if( presentationURL==null)
            presentationURL="http://"+baseURL;
        return buildXMLDescription(baseURL,
                                   friendlyName,
                                   manufacture,
                                   manufactureURL,
                                   modelDescription,
                                   modelName,
                                   uuid,
                                   serviceList,
                                   presentationURL);
    }
    
    // driver method, for testing
    public static void main(String[] argv){
        System.out.println("#args"+argv.length);
        if ( argv.length>1 ) { // run the server
            SSDP.VERB=2;
            System.out.println("Starting SSDP server");
            SSDP server=new SSDP(argv[0]);
            String descriptionxml=server.getXMLDescription(argv[0],"MindAffect","http://www.mindaffect.nl","Utopia-HUB BCI server","Utopia-HUB_v1.0","",null);
            // log the description.xml
            System.out.println("description.xml\n"+
                               descriptionxml);
            server.runServer(false); // ip4
            //server.runServer(true);  // ip6
        } else {
            SSDP.VERB=2;
            System.out.println("Starting SSDP discovery");
            // run the discovery client.
            try {
                SSDP.discover();
            } catch ( IOException ex ) {
                System.out.println("Exception running discovery");
                ex.printStackTrace();
            }
        }
        
    }
};
