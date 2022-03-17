package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

/**
 * General interface for all utopia Messages -- minimum methods they must provide
 * Also location for general information about the message format
 *
 *  LIst of used message IDs
    static final int STIMULUSEVENT      =(int)'E';
    static final int PREDICTEDTARGETPROB=(int)'P';
    static final int PREDICTEDTARGETDIST=(int)'F';
    static final int MODECHANGE         =(int)'M';
    static final int RESET              =(int)'R';
    static final int NEWTARGET          =(int)'N';
    static final int HEARTBEAT          =(int)'H';
    static final int SIGNALQUALITY      =(int)'Q';
    static final int LOG                =(int)'L';
    static final int SELECTION          =(int)'S';
    static final int TICKTOCK           =(int)'T';
    static final int TIMESTAMP2SAMPLE   =(int)'A';
    static final int DATAPACKET         =(int)'D';
    static final int SUBSCRIBE          =(int)'B';
    static final int OUTPUTSCORE        =(int)'O';
 */
public interface UtopiaMessage {
    int msgID();
    String msgName();
    void serialize(java.nio.ByteBuffer buf);
    //static UtopiaMessage deserialize(java.nio.ByteBuffer buf);
    // time-stamp getter/setter
    int gettimeStamp();
    void settimeStamp(int ts);
    int getVersion();

    public static final java.nio.ByteOrder UTOPIABYTEORDER    = java.nio.ByteOrder.LITTLE_ENDIAN;
    public static final java.nio.charset.Charset UTOPIACHARSET= java.nio.charset.Charset.forName("UTF-8");
};


