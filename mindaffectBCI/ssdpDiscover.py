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
    Return a list with tuples containing NIC and IPv4 address
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
    ssdpv4group = ("239.255.255.250", 1900)
    ssdpv6group= ("FF08::C", 1900)
    ssdpgroup = None

    msearchTemplate = "\r\n".join([
        'M-SEARCH * HTTP/1.1',
        'HOST: {0}:{1}',
        'MAN: "ssdp:discover"',
        'ST: {st}', 'MX: {mx}', '', ''])
    def __init__(self,servicetype="ssdp:all",discoverytimeout=3):
        self.servicetype=servicetype
        self.queryt=None
        self.sock=None

    def makeDiscoveryMessage(self,ssdpgroup,servicetype,timeout):
        msearchMessage = self.msearchTemplate.format(*ssdpgroup, st=servicetype, mx=timeout)
        if sys.version_info[0] == 3:
            msearchMessage = msearchMessage.encode("utf-8")
        return msearchMessage
    
    def initSocket(self, v6=False):
        # make the UDP socket to the multicast group with timeout
        if v6 and socket.has_ipv6:
            IPPROTO_IPV6 = socket.IPPROTO_IPV6 if hasattr(socket,'IPPROTO_V6') else 41
            self.ssdpgroup = self.ssdpv6group
            self.sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.sock.setsockopt(IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, 5)
                self.sock.setsockopt(IPPROTO_IPV6, socket.IPV6_MULTICAST_LOOP, 1)
                self.sock.setsockopt(IPPROTO_IPV6, socket.IPV6_MULTICAST_IF, 0)
            except:
                print("couldn't set socket options\n")
            self.sock.bind(('',self.ssdpgroup[1])) # bind port

            # request multicast group membership
            mreq = struct.pack("16s15s".encode('utf-8'), 
                                socket.inet_pton(socket.AF_INET6, self.ssdpgroup[0]), 
                                (chr(0) * 16).encode('utf-8'))
            self.sock.setsockopt(IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)

            if_idx=0
            self.sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_IF, struct.pack("I",if_idx))

        else:
            self.ssdpgroup = self.ssdpv4group
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 5)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

            self.sock.bind(('',self.ssdpgroup[1])) # bind port

            # request membership
            group = socket.inet_aton(self.ssdpgroup[0])

            # Set multicast interface -- only on unix!
            if_idx=0
            #ifi = socket.if_nametoindex(ifn)
            #ifis = struct.pack("I", ifi)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, struct.pack("I",if_idx))

            #local_ip = get_remote_ip()        
            #if local_ip is None:
            #   local_ip=get_local_ip()
            #print("Trying local ip: {}".format(local_ip))
            #mreq = group + socket.inet_aton(local_ip)

            mreq = struct.pack('4sl', group, socket.INADDR_ANY)

            # add to multi-cast group to get responses
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        
    def discover(self,timeout=.001,querytimeout=5,v6=False):
        '''auto-discover the utopia-hub using ssdp discover messages,
           timeout is time to wait for a response.  query timeout is time between re-sending the ssdp-discovery query message.'''
        # make and send the discovery message
        if self.sock is None:
            try : 
                self.initSocket(v6)
            except socket.error :
                print("Couldnt init socket!")
                return ()

        # re-send the ssdp query
        if self.queryt is None or self.queryt+querytimeout < time.time():
            # build discovery message for this service/group
            self.msearchMessage=self.makeDiscoveryMessage(self.ssdpgroup, self.servicetype, querytimeout)

            print("Sending query message:\n%s"%(self.msearchMessage))
            try:
                self.sock.sendto(self.msearchMessage, self.ssdpgroup)
                self.queryt = time.time()
            except socket.error as ex: # abort if send fails
                print("send error")
                print(ex)
                return ()

        # wait for the specified time for a response
        t0 = time.time()
        ttg = timeout
        responses = []
        while time.time() < t0 + timeout:
            location, rsp = self.wait_msearch_response(ttg)
            if location is not None:
                responses.append(location)
            ttg = t0 + timeout - time.time()

        # wait for responses to our query
        return responses

    def wait_msearch_response(self,timeout):
        self.sock.settimeout(timeout)
        print("Waiting responses")
        try:
            rsp,addr=self.sock.recvfrom(8192)
        except socket.timeout:
            print("Socket timeout")
            return (None,None)
        except socket.error as ex :
            print("Socket error" + str(ex)) 
            return (None,None)

        if rsp == self.msearchMessage:
            print("Response was query message!")
            return (None,None)


        # got a response, parse it...           
        rsp=rsp.decode('utf-8')
        print("Got response from : %s\n%s\n"%(addr,rsp))

        # guard for recieving our own query message!
        #if "M-SEARCH" in rsp:

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
        return (location,rsp)


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
    disc=ssdpDiscover("utopia/1.1")
    while True:
        d0 = time.time()
        resp = disc.discover(timeout=.001,v6=False)
        dt = time.time()-d0
        print("Discover time: " + str(dt))
        if len(resp)==0 :
            print("No new responses")
        else :
            print(resp)
            break
        # sleep to represent doing other work..
        time.sleep(1)

def discoverOrIPscan(port:int = 8400, timeout_ms:int = 15000):
    disc=ssdpDiscover("utopia/1.1")
    hosts = disc.discover(timeout=timeout_ms/1000.0,v6=False)
    
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