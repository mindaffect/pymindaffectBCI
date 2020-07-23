#!/usr/bin/env python3
import socket
import sys
import time
import re
import struct

def ip_is_local(ip_string):
    """
    Uses a regex to determine if the input ip is on a local network. Returns a boolean. 
    It's safe here, but never use a regex for IP verification if from a potentially dangerous source.
    """
    combined_regex = "(^10\.)|(^172\.1[6-9]\.)|(^172\.2[0-9]\.)|(^172\.3[0-1]\.)|(^192\.168\.)"
    return re.match(combined_regex, ip_string) is not None # is not None is just a sneaky way of converting to a boolean

def get_ip_address( NICname ):
    import fcntl
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', NICname[:15].encode("UTF-8"))
    )[20:24])

def nic_info():
    """
    Return a list with tuples containing NIC and IPv4
    """
    nic = []
    for ix in socket.if_nameindex():
        name = ix[1]
        ip = get_ip_address( name )

        nic.append( (name, ip) )
    return nic    

def get_all_ips():
    try:
        ips = [ i[1] for i in nic_info() ]
    except:
        # fall back on getaddrinfo
        ips = [ x[4][0] for x in socket.getaddrinfo(socket.gethostname(), 80) ]
    return ips

def get_local_ip():
    # socket.getaddrinfo returns a bunch of info, so we just get the IPs it returns with this list comprehension.
    local_ips = [ i for i in get_all_ips() if ip_is_local(i) ]
    # prefer one with internet access..
    print("all local ips: {}".format(local_ips))
    local_ip = local_ips[0] if len(local_ips) > 0 else '127.0.0.1'
    return local_ip

def get_remote_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except socketerror:
        ip = None
    return ip

class ssdpDiscover :
    ssdpgroup = ("239.255.255.250", 1900)
    msearchTemplate = "\r\n".join([
        'M-SEARCH * HTTP/1.1',
        'HOST: {0}:{1}',
        'MAN: "ssdp:discover"',
        'ST: {st}', 'MX: {mx}', '', ''])
    def __init__(self,servicetype="ssdp:all",discoverytimeout=5):
        self.servicetype=servicetype
        self.queryt=None
        self.sock=None
        self.msearchMessage=self.makeDiscoveryMessage(servicetype,discoverytimeout)

    def makeDiscoveryMessage(self,servicetype,timeout):
        msearchMessage = self.msearchTemplate.format(*self.ssdpgroup, st=servicetype, mx=timeout)
        if sys.version_info[0] == 3:
            msearchMessage = msearchMessage.encode("utf-8")
        return msearchMessage
    
    def initSocket(self):
        # make the UDP socket to the multicast group with timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)#, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self.sock.bind(('',self.ssdpgroup[1])) # bind port
        #local_ip = get_remote_ip()
        
        #if local_ip is None:
        #    local_ip=get_local_ip()
        #print("Trying local ip: {}".format(local_ip))
        #membership_request = socket.inet_aton(self.ssdpgroup[0]) + socket.inet_aton(local_ip)
        ### Send add membership request to socket
        #self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership_request)

        # request membership
        group = socket.inet_aton(self.ssdpgroup[0])
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        
    def discover(self,timeout=.001,querytimeout=5):
        '''auto-discover the utopia-hub using ssdp discover messages,
           timeout is time to wait for a response.  query timeout is time between re-sending the ssdp-discovery query message.'''
        # make and send the discovery message
        if self.sock is None:
            try : 
                self.initSocket()
            except socket.error :
                print("Couldnt init socket!")
                return ()

        # re-send the ssdp query
        if self.queryt is None or self.queryt+querytimeout < time.time():
            print("Sending query message:\n%s"%(self.msearchMessage))
            try:
                self.sock.sendto(self.msearchMessage, self.ssdpgroup)
                self.queryt = time.time()
            except socket.error as ex: # abort if send fails
                print("send error")
                print(ex)
                return ()

        # wait for responses to our query
        self.sock.settimeout(timeout)
        print("Waiting responses")
        try:
            rsp,addr=self.sock.recvfrom(8192)
        except socket.timeout:
            print("Socket timeout")
            return ()
        except socket.error as ex :
            print("Socket error" + str(ex)) 
            return ()

        # got a response, parse it...
        responses=[]
        rsp=rsp.decode('utf-8')
        print("Got response from : %s\n%s\n"%(addr,rsp))
        # does the response contain servertype, if so then we match
        location=None
        if self.servicetype is None or self.servicetype in rsp :
            print("Response matches server type: %s"%(self.servicetype))
            # use the message source address as default location
            location=addr[0] if hasattr(addr,'__iter__') else addr 
            # extract the location or IP from the message
            for line in rsp.split("\r\n"): # cut into lines
                tmp=line.split(":",1) # cut line into key/val
                # is self the key we care about -> then record the value
                if len(tmp)>1 and tmp[0].lower()=="LOCATION".lower() :
                    location=tmp[1].strip()
                    # strip http:// xxxx /stuff
                    if location.startswith("http://"):
                        location=location[7:] # strip http
                    if '/' in location :
                        location=location[:location.index('/')] # strip desc.xml
                    print("Got location: {}".format(location))
                    break # done with self response
            # add to the list of possible servers
            print("Loc added to response list: {}".format(location))
            responses.append(location)
        return responses

def ipscanDiscover(port:int, ip:str=None, timeout=.5):
    ''' scan for service by trying all 255 possible final ip-addresses'''
    if ip is None:
        ip = get_remote_ip()
    if ip is None:
        #    local_ip=get_local_ip()
        ips = get_all_ips()
        # prefer non local-host IP if there is one.
        nonlocalip = [ i for i in ips if not i.startswith('127.') ]
        print("nonlocalhost={}".format(nonlocalip))        
        ip = nonlocalip[0] if nonlocalip else ips[0]
        
    ipprefix = ".".join(ip.split('.')[:3])
    hosts = []
    for postfix in range(255):
        ip = "{}.{:d}".format(ipprefix,postfix)
        try:
            print("Trying: {}:{}".format(ip,port))
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((ip,port))            
            print("Connected")
            hosts.append(ip)
            sock.close()
        except socket.error as ex:
            pass
    return hosts


def testIncrementalScanning():
    disc=ssdpDiscover("utopia/1.1");
    while True:
        d0 = time.time()
        resp = disc.discover(timeout=.001)
        dt = time.time()-d0
        print("Discover time: " + str(dt))
        if len(resp)==0 :
            print("No new responses")
        else :
            print(resp)
            break;
        # sleep to represent doing other work..
        time.sleep(1)

def discoverOrIPscan(port:int = 8400, timeout_ms:int = 5000):
    disc=ssdpDiscover("utopia/1.1");
    hosts = disc.discover(timeout=timeout_ms/1000.0)
    
    if hosts is None or len(hosts) == 0:
        hosts=ipscanDiscover(8400)
    print("got hosts: {}".format(hosts))
    return hosts

if __name__=="__main__":
    try:
        print( nic_info() )
    except:
        pass
    
    discoverOrIPscan()