#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__WIN32__) || defined(__WIN64__)
  #include<winsock2.h>
  typedef unsigned long in_addr_t;
#else
  #include <netdb.h> /* getprotobyname */
  #include <arpa/inet.h>
  #include <sys/socket.h>
  #include <netinet/tcp.h>
#endif
#include <inttypes.h>
#include <unistd.h>
#include <time.h>

#include "utopiaclient.h"
const int BUFSIZE=8192;
const int DEFAULTUTOPIAPORT=8400;
const int VERBOSITY=0;

int connect2utopia(char *host, int port){
  char protoname[] = "tcp";
  struct protoent *protoent;
  in_addr_t in_addr;
  in_addr_t server_addr;
  int sockfd;
  struct hostent *hostent;
  /* This is the struct used by INet addresses. */
  struct sockaddr_in sockaddr_in;

  if ( port<=0 ) port=DEFAULTUTOPIAPORT;

  /* Note on WINDOWs you *must* do this for socket functions to work */
  // startup WinSock in Windows
#if defined(__WIN32__) || defined(__WIN64__)
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(1,1), &wsa_data);
#endif    

  /* Get socket. */
  protoent = getprotobyname(protoname);
  if (protoent == NULL) {
    perror("getprotobyname");
    exit(-1);
  }
  sockfd = socket(AF_INET, SOCK_STREAM, protoent->p_proto);
  if (sockfd == -1) {
    perror("socket");
    exit(-1);
  }

  /*optinally disable nagel as we send lots of small packets?*/
  int flag = 1;
  setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *) &flag, sizeof(int));

  /* Prepare sockaddr_in. */
  hostent = gethostbyname(host);
  if (hostent == NULL) {
    fprintf(stderr, "error: gethostbyname(\"%s\")\n", host);
    exit(-1);
  }
  in_addr = inet_addr(inet_ntoa(*(struct in_addr*)*(hostent->h_addr_list)));
  if (in_addr == (in_addr_t)-1) {
    fprintf(stderr, "error: inet_addr(\"%s\")\n", *(hostent->h_addr_list));
    exit(EXIT_FAILURE);
  }
  sockaddr_in.sin_addr.s_addr = in_addr;
  sockaddr_in.sin_family = AF_INET;
  sockaddr_in.sin_port = htons(port);

  /* Do the actual connection. */
  if (connect(sockfd, (struct sockaddr*)&sockaddr_in, sizeof(sockaddr_in)) == -1) {
    perror("connect");
    return -1;
  }
  fprintf(stdout,"Connected to %s:%d\n",host,port);
  return sockfd;
}

/* millisecond time stamp, relative to first call */
struct timespec utopiaclientt0={0};
int getTimeStamp(){
  struct timespec curtime;
  clock_gettime(CLOCK_REALTIME, &curtime);
  if(utopiaclientt0.tv_sec==0) utopiaclientt0=curtime;
  int timeStamp = 1000*(curtime.tv_sec-utopiaclientt0.tv_sec) + (curtime.tv_nsec-utopiaclientt0.tv_nsec)/1000000;
  return timeStamp;
}

int sendMessage(int sockfd, struct RawMessage *msg){
  int buflen=BUFSIZE;
  char msgbuffer[buflen];
  size_t msglen;

  /* serialize the message into a byte stream */
  msglen = serializeMessage(msgbuffer,buflen,msg);
  /* log the serialized message*/
  if ( VERBOSITY>2 ) fprintf(stdout,"%.s\n",msglen,msgbuffer);
  
  if ( msglen>0 ) {
    /* send to the utopia-hub */
    if (send(sockfd, msgbuffer, msglen, 0) == -1) {
      perror("sendMessage:send");
      exit(EXIT_FAILURE);
    }
  }
  return msglen;
}

struct RawMessage* getMessages(int sockfd){
  int nbytes_read=0;
  char buffer[BUFSIZ];  
  while ((nbytes_read = recv(sockfd, buffer, BUFSIZ,0)) > 0) {
    send(STDOUT_FILENO, buffer, nbytes_read, 0);
    if (buffer[nbytes_read - 1] == '\n') {
      fflush(stdout);
      break;
    }
  }
  return 0;
}

void subscribeNone(int sockfd){
   struct Subscribe msg;
   msg.timeStamp=getTimeStamp();
    msg.messageIDs="";
    msg.nmessageIDs=0;
    /* setup the header */
    struct RawMessage rawmsg;
    rawmsg.msgID=SUBSCRIBEMSGID;
    rawmsg.version=0;
    rawmsg.payload=&msg;
    /* send the message */
    sendMessage(sockfd,&rawmsg);
}

int serializeMessage(char* buffer, int buflen, struct RawMessage *msg){  
  int16_t payloadlen=0;
  /* do the fixed header parts */
  buffer[0]=msg->msgID;
  buffer[1]=msg->version;
    
  /* serialize in the payload, and get it's size */
  switch ( msg->msgID ){
  case DATAPACKETMSGID: /* dataPacket */
    payloadlen = serializeDataPacket(buffer+HEADERSIZE,buflen-HEADERSIZE,
                                     (struct DataPacket*)(msg->payload));
    break;
  case SUBSCRIBEMSGID: /* subscribe */
    payloadlen = serializeSubscribe(buffer+HEADERSIZE,buflen-HEADERSIZE,
                                    (struct Subscribe*)msg->payload);
    break;
  case NEWTARGETMSGID: /* new target */
    payloadlen = serializeNewTarget(buffer+HEADERSIZE,buflen-HEADERSIZE,
                                    (struct NewTarget*)msg->payload);
    break;
  case HEARTBEATMSGID: /* heartbeat */
    payloadlen = serializeHeartbeat(buffer+HEADERSIZE,buflen-HEADERSIZE,
                                    (struct Heartbeat*)msg->payload);
    break;
  case SELECTIONMSGID: /* selection */
    payloadlen = serializeSelection(buffer+HEADERSIZE,buflen-HEADERSIZE,
                                    (struct Selection*)msg->payload);
    break;
  default:
    payloadlen=-1;
  }
  if ( payloadlen < 0 ) {
    fprintf(stdout,"Unsupported message type %c, ignored\n",msg->msgID);
    return -1;
  }
  /* set the size of the payload in the header */
  memcpy(buffer+2, &(payloadlen), sizeof(int16_t));
  return HEADERSIZE+(int)payloadlen;
}


int serializeDataPacket(char *buffer, int buflen, struct DataPacket *msg){
  size_t payloadlen=0;
  size_t fieldsize=0;
  /* timeStamp */
  int timeStamp=msg->timeStamp;
  if( timeStamp<0 ) timeStamp=getTimeStamp();
  fieldsize=sizeof(timeStamp);
  if( buflen<payloadlen+fieldsize ) { perror("sd1: buff to small"); return -1; }
  memcpy(buffer+payloadlen, &(timeStamp), fieldsize);
  payloadlen+=fieldsize;
  /* nsamples */
  fieldsize=sizeof(msg->nsamples);
  if( buflen<payloadlen+fieldsize ) { perror("sd2: buff to small"); return -1;}
  memcpy(buffer+payloadlen, &(msg->nsamples), fieldsize);
  payloadlen+=fieldsize;
  /* sammples */
  fieldsize=sizeof(msg->samples[0])*msg->nsamples*msg->nchannels;
  if( buflen<payloadlen+fieldsize ) { perror("sd3: buff to small"); return -1;}
  memcpy(buffer+payloadlen, msg->samples,  fieldsize);
  payloadlen+=fieldsize;
  return payloadlen;  
}

int serializeSelection(char *buffer, int buflen, struct Selection *msg){
  size_t payloadlen=0;
  size_t fieldsize=0;
  /* timeStamp */
  fieldsize=sizeof(msg->timeStamp);
  if( buflen<payloadlen+fieldsize ) { perror("ss1: buff to small"); return -1;}
  memcpy(buffer+payloadlen, &(msg->timeStamp), fieldsize);
  payloadlen+=fieldsize;
  /* objID */
  fieldsize=sizeof(msg->objID);
  if( buflen<payloadlen+fieldsize ) { perror("ss2: buff to small"); return -1;}
  memcpy(buffer+payloadlen, &(msg->objID), fieldsize);
  payloadlen+=fieldsize;
  return payloadlen;
}

int serializeSubscribe(char *buffer, int buflen, struct Subscribe* msg){
  size_t payloadlen=0;
  size_t fieldsize=0;
  /* timeStamp */
  int timeStamp=msg->timeStamp;
  if( timeStamp<0 ) timeStamp=getTimeStamp();
  fieldsize=sizeof(timeStamp);
  if( buflen<payloadlen+fieldsize ) { perror("sd1: buff to small"); return -1; }
  memcpy(buffer+payloadlen, &(timeStamp), fieldsize);
  payloadlen+=fieldsize;
  /* messageIDs */
  fieldsize=sizeof(msg->messageIDs[0])*msg->nmessageIDs;
  if( buflen<payloadlen+fieldsize ) { perror("sb2: buff to small"); return -1;}
  memcpy(buffer+payloadlen, msg->messageIDs,  fieldsize);
  payloadlen+=fieldsize;
  return payloadlen;  
}

int serializeNewTarget(char *buffer, int buflen, struct NewTarget *msg){
  size_t payloadlen=0;
  size_t fieldsize=0;
  /* timeStamp */
  int timeStamp=msg->timeStamp;
  if( timeStamp<0 ) timeStamp=getTimeStamp();
  fieldsize=sizeof(timeStamp);
  if( buflen<payloadlen+fieldsize ) { perror("sd1: buff to small"); return -1; }
  memcpy(buffer+payloadlen, &(timeStamp), fieldsize);
  payloadlen+=fieldsize;
  return payloadlen;
}

int serializeHeartbeat(char *buffer, int buflen, struct Heartbeat *msg){
  size_t payloadlen=0;
  size_t fieldsize=0;
  /* timeStamp */
  int timeStamp=msg->timeStamp;
  if( timeStamp<0 ) timeStamp=getTimeStamp();
  fieldsize=sizeof(timeStamp);
  if( buflen<payloadlen+fieldsize ) { perror("sd1: buff to small"); return -1; }
  memcpy(buffer+payloadlen, &(timeStamp), fieldsize);
  payloadlen+=fieldsize;
  return payloadlen;
}



void testSending(){ 

  int sockfd = connect2utopia("localhost",8400);

  struct RawMessage rawmsg;

  /* send subscribe */
  {
    /* make the playload */
    struct Subscribe msg;
    msg.timeStamp=getTimeStamp();
    msg.messageIDs="NH";
    msg.nmessageIDs=2;
    /* setup the header */
    rawmsg.msgID=SUBSCRIBEMSGID;
    rawmsg.version=0;
    rawmsg.payload=&msg;
    /* send the message */
    sendMessage(sockfd,&rawmsg);
  }

  /* send datapacket */
  {
    struct DataPacket msg;
    msg.timeStamp=getTimeStamp();
    msg.nsamples=5;
    msg.nchannels=4;
    msg.samples = (float*)malloc(msg.nsamples*msg.nchannels*sizeof(float));
    int i;
    for ( i=0; i<msg.nsamples*msg.nchannels; i++){
      msg.samples[i] = i;
    }
    /* setup the raw message envelope */
    rawmsg.msgID=DATAPACKETMSGID;
    rawmsg.version=0;
    rawmsg.payload=&msg;
    /* send the message */
    sendMessage(sockfd,&rawmsg);    
  }

  /* send heartbeat */
  {
    struct Heartbeat msg;
    msg.timeStamp=getTimeStamp();
    /* setup the raw message envelope */
    rawmsg.msgID=HEARTBEATMSGID;
    rawmsg.version=0;
    rawmsg.payload=&msg;
    /* send the message */
    sendMessage(sockfd,&rawmsg);    
  }
}
