package nl.ma.utopia;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import nl.fcdonders.fieldtrip.bufferserver.BufferServer;
import nl.fcdonders.fieldtrip.bufferserver.FieldtripBufferMonitor;
import nl.fcdonders.fieldtrip.bufferserver.data.Data;
import nl.fcdonders.fieldtrip.bufferserver.exceptions.DataException;
import nl.fcdonders.fieldtrip.bufferserver.network.Request;
import nl.ma.utopiaserver.UtopiaClient;
import nl.ma.utopiaserver.messages.Subscribe;
import nl.ma.utopiaserver.messages.DataPacket;
import java.nio.ByteBuffer;
import java.io.IOException;

/**
 * Main class for the forwarding of utopia-messages to fieldtrip-buffer messages and back-again
 */
public class FT2Utopia implements Runnable, FieldtripBufferMonitor {
    public static int VERBOSITY=1; // the level of debugging verbosity to use. lower=quieter
    public static String TAG="ft2utopia:";
    public static int LOGLEVEL=1; 
    
    public static String usage="ft2utopia utopiahost:utopiaport buffport";
    
    /**
     * Driver method to startup the forwarding process
     */
	 public static void main(String[] args) {
        System.out.println(usage);
	
      String utopiahost = "-";
      int utopiaport   = 8400;
      if (args.length>=1) {
          utopiahost = args[0];
          int sep = utopiahost.indexOf(':');
          if ( sep>0 ) {
               utopiaport=Integer.parseInt(utopiahost.substring(sep+1,utopiahost.length()));
               utopiahost=utopiahost.substring(0,sep);
          }			
      }

      int buffport   = DEFAULTBUFFERPORT;
      if (args.length>=2) {
          buffport=Integer.parseInt(args[1]);
      }
        System.out.println(TAG+" buffport="+buffport);

      System.out.println(TAG+" VERBOSITY="+VERBOSITY);
      try {
          FT2Utopia ft2u = new FT2Utopia(utopiahost, utopiaport, buffport);
          // Run the stuff
          ft2u.run();
      } catch ( IOException ex ){

      }
	}
    
    /**
     * connection to the field-trip buffer, for recieving messages and time-stamp tracking
     */
    BufferServer S;
    public final static int DEFAULTBUFFERPORT=1972;
    /**
     * Locally running utopia-message server
     */
    UtopiaClient U;

    public FT2Utopia(final String utopiahost, int utopiaport, int buffport) throws IOException {
        // BufferServer
        // start the buffer server if path is set
        S = new BufferServer(buffport);
        S.addMonitor(this); // callbacks for when stuff happens
        System.out.println("Buffer Server on : " + buffport);

        // Utopia Server
        U = new UtopiaClient();
        U.connect(utopiahost,utopiaport);
        t0=U.gettimeStamp();
        nextLogTime_ms=t0;
        U.sendMessage(new Subscribe(U.gettimeStamp(),"")); // no-subscriptions
    }
    
    public void run(){
        // run the server
        S.run(); 
    }


    int t0;
    int nextLogTime_ms;
    int nBlk=0, nSamp=0;
    int utopia_exceptions=0;

    @Override
	public void clientPutSamples(final int count, final int clientID,
			final int diff, final long time) {
            /* call-back when got new samples in the buffer */
        if ( VERBOSITY > 2 ) {
            System.out.print(" Client "
                    + clientID + " added " + diff
                    + " samples, total now " + count + "\r");
        }

        // get the samples from the ring-buffer
        Data buffdata=null;
        try {
            buffdata = S.getDataStore().getData(new Request(count - diff-1, count-1));
        } catch ( DataException ex ){
            System.out.println("Exception getting data! " + ex);
        }

        if ( buffdata != null ) {
            // convert from byte-array to float
            float[][] samples = getFloatData(buffdata);
            // convert to utopia data packet
            DataPacket dp = new DataPacket(U.gettimeStamp(), samples);
            if (VERBOSITY > 2) System.out.println(TAG + "Sending:" + buffdata);
            try {
                U.sendMessage(dp);
                U.getNewMessages(0); // flush incomming message stream
            } catch ( IOException ex ){
                System.out.println("Exception sending data to utopia!"+ex);
                utopia_exceptions = utopia_exceptions + 1;
                if ( utopia_exceptions > 50){
                    System.out.println("Looks like utopia is dead.  Stopping..");
                    System.exit(-1);
                }
            }
            nBlk += 1;
            nSamp += dp.nsamples;
        }
        // logging
        int t=U.gettimeStamp();
        if( t > nextLogTime_ms ){
            System.out.print("\n" + nSamp + " " + nBlk + " " + ((t-t0)/1000f) + " " 
                                  + nSamp*1000/((t-t0)+1) + " (samp,blk,sec,hz)\n");
            System.out.flush();
            nextLogTime_ms = t + 2000;
        }            
	}


	class DataType {
        public static final int UNKNOWN = -1;
        public static final int CHAR    = 0;
        public static final int UINT8   = 1;
        public static final int UINT16  = 2;
        public static final int UINT32  = 3;
        public static final int UINT64  = 4;
        public static final int INT8    = 5;
        public static final int INT16   = 6;
        public static final int INT32   = 7;
        public static final int INT64   = 8;
        public static final int FLOAT32 = 9;
        public static final int FLOAT64 = 10; };

    float[][] getFloatData(Data dd){
		int nSamples = dd.nSamples;
		int nChans = dd.nChans;
        byte [][][] bsamp = dd.data;
        ByteBuffer buf;
		
		float[][] data = new float[nSamples][nChans];
		
		switch (dd.dataType) {
			case DataType.INT8:
				for (int i=0;i<nSamples;i++) {
					for (int j=0;j<nChans;j++) {
						data[i][j] = (float) ByteBuffer.wrap(dd.data[i][j]).order(dd.order).get();
					}
				}
				break;
			case DataType.INT16:
				for (int i=0;i<nSamples;i++) {
					for (int j=0;j<nChans;j++) {
						data[i][j] = (float) ByteBuffer.wrap(dd.data[i][j]).order(dd.order).getShort();
					}
				}
				break;
			case DataType.INT32:
				for (int i=0;i<nSamples;i++) {
					for (int j=0;j<nChans;j++) {
						data[i][j] = (float) ByteBuffer.wrap(dd.data[i][j]).order(dd.order).getInt();
					}
				}
				break;
			case DataType.FLOAT32:
                for (int i=0;i<nSamples;i++) {
                    for (int j=0;j<nChans;j++) {
                        data[i][j] = (float) ByteBuffer.wrap(dd.data[i][j]).order(dd.order).getFloat();
                    }
                }
				break;
			case DataType.FLOAT64:
				for (int i=0;i<nSamples;i++) {
					for (int j=0;j<nChans;j++) {
						data[i][j] = (float) ByteBuffer.wrap(dd.data[i][j]).order(dd.order).getDouble();
					}
				}
				break;
			default:
				System.out.println("Not supported yet - returning zeros.");
		}	
		return data;
	}

	// Null versions for the rest of the methods
    public void clientClosedConnection(int clientID, long time){}
    public void clientContinues(int clientID, long time){}
    public void clientError(int clientID, int errorType, long time){}
    public void clientFlushedData(int clientID, long time){}
    public void clientFlushedEvents(int clientID, long time){}
    public void clientFlushedHeader(int clientID, long time){}
    public void clientGetEvents(int count, int clientID, long time){}
    public void clientGetHeader(int clientID, long time){}
    public void clientGetSamples(int count, int clientID, long time){}
    public void clientOpenedConnection(int clientID, String adress, long time){}
    public void clientPolls(int clientID, long time){}
    public void clientPutEvents(int count, int clientID, int diff, long time){}
    public void clientPutHeader(int dataType, float fSample, int nChannels, int clientID, long time){}
    public void clientWaits(int nSamples, int nEvents, int timeout, int clientID, long time){}
};
