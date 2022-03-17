package nl.ma.utopiaserver.messages;
/*
 * Copyright (c) MindAffect B.V. 2018
 * For internal use only.  Distribution prohibited.
 */

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import nl.ma.utopiaserver.ClientException;

/**
 * the DATA utopia message, which gives the stimulus state for a set of currently active stimulus-object identifiers.
 */
public class DataPacket implements UtopiaMessage {
    public static final int MSGID         =(int)'D';
    public static final String MSGNAME    ="DATAPACKET";
    /**
     * get the unique message ID for this message type
     */
    public int msgID(){ return MSGID; }
    /**
     * get the unique message name, i.e. human readable name, for this message type
     */
    public String msgName(){ return MSGNAME; }

    public int timeStamp;
    /**
     * get the time-stamp for this message 
     */
    public int gettimeStamp(){return this.timeStamp;}
    /**
     * set the time-stamp information for this message.
     */
    public void settimeStamp(int ts){ this.timeStamp=ts; }
    /**
     * get the version of this message
     */
    public int getVersion(){return 0;}
    /**
     * The raw samples obtained from the buffer
     */
    public float[] samples;
    public int nchannels;
    public int nsamples;
    
    public DataPacket(final int timeStamp, final int nsamples, final int nchannels){
        this.timeStamp=timeStamp;
        this.nsamples =nsamples;
        this.nchannels=nchannels;
        this.samples  =new float[nsamples*nchannels];        
    }
    public DataPacket(final int timeStamp, final int nsamples, final int nchannels, final float[] samples){
        this.timeStamp=timeStamp;
        this.nsamples =nsamples;
        this.nchannels=nchannels;
        this.samples  =new float[nsamples*nchannels];
        System.arraycopy(samples,0,this.samples,0,nsamples*nchannels);        
    }
    public DataPacket(final int timeStamp, final float[][] samples){
        this.timeStamp=timeStamp;
        this.nsamples = samples.length;
        this.nchannels= samples[0].length;
        this.samples  =new float[samples.length*samples[0].length];
        for ( int t=0, i=0; t<samples.length; t++){
            System.arraycopy(samples[t],0,this.samples,i,samples[t].length);
            i+=samples[t].length;
        }
    }    


    /**
     * convert the internal 1-d array representation to a 2-d one.
     */ 
    public float[][] getSamples(){
        float[][] samp2d=new float[nsamples][nchannels];
        for ( int t=0, i=0; t<nsamples; t++){
            for( int c=0; c<nchannels; c++, i++){
                samp2d[t][c]=samples[i];
            }
        }
        return samp2d;
    }
    
    /**
     * deserialize a byte-stream to create an instance of this class 
     * @throws exception if the byte-stream does not contains a validly encoded DATA
     */ 
    public static DataPacket deserialize(final ByteBuffer buffer, int version)
        throws ClientException {

        buffer.order(UTOPIABYTEORDER);
        // get the timestamp
        final int timeStamp = buffer.getInt();
        // get number channels per time-point
        final int nSamples = buffer.getInt();
        final int nChannels = (int)buffer.remaining()/4/nSamples;
       
        // Check if size and the number of bytes in the buffer match
        if ( buffer.remaining() != nChannels*nSamples*4) {
            System.err.println("Number samples not multiple of number of channels!");
        }
        
       // extract into 2 arrays, 1 for the objIDs and one for the state      
       // Transfer bytes from the buffer into a nSamples*nChans*nBytes array;
       float[] samples   = new float[nSamples*nChannels];
       for (int t = 0; t < nSamples*nChannels; t++) {
           samples[t]=buffer.getFloat();
           //samples[t]=new float[nChannels];
           //for (int i = 0; i < nChannels; i++) {
           //    samples[t][i]    = buffer.getFloat();
           //}
       }
       return new DataPacket(timeStamp,nSamples,nChannels,samples);
    }
    public static DataPacket deserialize(final ByteBuffer buffer)
        throws ClientException {
        return deserialize(buffer,0);
    }
    /**
     * serialize this instance into a byte-stream in accordance with the message spec. 
     */
    public void serialize(final ByteBuffer buf) {
        buf.order(UTOPIABYTEORDER);
        buf.putInt(timeStamp);
        buf.putInt(nsamples);//samples[0].length); // number channels-per-sample
        for( int t=0; t<nsamples*nchannels; t++ ){
            //for( int i=0; i<samples[t].length; i++) {
                buf.putFloat(samples[t]);
            //}
        }
    }
    
    public String toString() {
	String str= "t:" + msgName() + " ts:" + timeStamp ;
	str = str + " v[" + + nchannels + "x" + nsamples + "]:";
	for( int t=0, i=0; t<nsamples; t++ ){
	    str = str + "[";
	    for( int c=0; c<nchannels; c++, i++) {
		str = str + samples[i] +",";
	    }
	    str=str+"]";
	}
	return str;
    }
};
