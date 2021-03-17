#ifndef _UTOPIACLIENTH_
#define _UTOPIACLIENTH_

#include <stdint.h>

/*----------------------------- message structures ---------------------*/
struct RawMessage {
  char msgID;
  uint8_t version;
  uint16_t msglength;
  void *payload;
};
const int HEADERSIZE= 1+1+2;

struct DataPacket {
  /* msgID = D */
  uint32_t timeStamp;
  uint32_t nsamples;
  uint32_t nchannels;
  float* samples;
};
#define DATAPACKETMSGID 'D'

struct Subscribe {
  /* msgID = B */
  uint32_t timeStamp;
  char* messageIDs;
  int nmessageIDs;
};
#define SUBSCRIBEMSGID 'B'

struct NewTarget {
  /* msgID = N */
  uint32_t timeStamp;
};
#define NEWTARGETMSGID 'N'

struct Heartbeat {
  /* msgID = H */
  uint32_t timeStamp;
};
#define HEARTBEATMSGID 'H'

struct Selection {
  uint32_t timeStamp;
  char objID;
};
#define SELECTIONMSGID 'S'


/*-------------------------- function prototypes ------------------------*/
int connect2utopia(char *host, int port);

int getTimeStamp();
void subscribeNone(int sockfd);

int sendMessage(int sockfd, struct RawMessage *msg);
int serializeMessage(char* buffer, int buflen, struct RawMessage *msg);
int serializeDataPacket(char *buffer, int buflen, struct DataPacket *msg);
int serializeSelection(char *buffer, int buflen, struct Selection *msg);
int serializeSubscribe(char *buffer, int buflen, struct Subscribe* msg);
int serializeNewTarget(char *buffer, int buflen, struct NewTarget *msg);
int serializeHeartbeat(char *buffer, int buflen, struct Heartbeat *msg);

struct RawMessage* getMessages(int sockfd);
#endif
