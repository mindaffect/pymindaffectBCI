#ifndef _UTOPIACLIENTH_
#define _UTOPIACLIENTH_

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

#endif
