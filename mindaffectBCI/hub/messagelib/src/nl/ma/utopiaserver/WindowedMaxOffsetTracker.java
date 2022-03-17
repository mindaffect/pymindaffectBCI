/*
 * Copyright (C) 2014, Jason Farquhar
 * Donders Institute for Donders Institute for Brain, Cognition and Behaviour,
 * Centre for Cognition, Radboud University Nijmegen,
 */
package nl.ma.utopiaserver; // test
import java.util.Deque;
import java.util.ArrayDeque;

/**
 * Windowed running maximum tracking from an input stream of values.
 */
public class WindowedMaxOffsetTracker implements ClockAlignment {
	 public double m, b;         // fit, scale and offset
	 double sErr=0;              // running estimate of the est-true Y error
    double alpha=.999;
    WindowedMaxTracker windowedMax;
    
	 public WindowedMaxOffsetTracker(){ this(60*1000); } // default to 1min window size
    public WindowedMaxOffsetTracker(int windowSize){
        windowedMax = new WindowedMaxTracker(windowSize);
        reset();
	 }
    /**
     * set the decay constant for the weighting, from a given half-life.
     * @param hl - the half life to use
     */
	 public void setHalfLife(int hl){
        windowedMax.setWindowSize(hl);
	 }

    /**
     * reset the regression information to initial (memory free) state
     */
	 public void reset(){
        windowedMax.reset();
		  m=1; b=0;
	 }

    /**
     * add a new point to the regression.
     * @param X - the new points X value
     * @param Y - the new points Y value
     */
	 public void addPoint(double X, double Y) {
        double diff=Y-X;
        //System.out.println("X="+X+"diff="+diff);
        windowedMax.addPoint(X,diff);
        b = windowedMax.getMaxVal();        
        double estErr = Math.abs(Y - getY(X));
        sErr = alpha*sErr + (1.0-alpha)*estErr; // moving average error estimate
	 }

    /**
     * get the x coordinate from the regression given the Y
     * @param Y - the y to get the X for
     */    
	 public double getX(double Y){  return (Y-b)/m; }
    /**
     * get the Y coordinate from the regression given the x
     * @param X - the y to get the Y for
     */    
	 public double getY(double X){  return m*X + b; }	 
    /**
     * get the current estimate of the error in the Y estimate given the X
     */
	 public double getYErr(){       return sErr; }

    public String toString(){
        return "Y= " + m + "*X + " + b + " (+/- " + getYErr() + ") ";
    }


    /**
     * Inner class to implement the windowed max tracker using a dequeue
     * deque to hold the set of non-dominated values in this window.
     * A value is non-dominated if there are no smaller values
     * in front of it in the queue. (i.e. if the queue remains in sorted order)
     * Thus; the 1st entry in the queue is the largest value in the window
     * and the 2nd entry next largest *with only smaller values before it* etc.
     */
     static final class WindowedMaxTracker {
         public static final class KeyVal {// struct to hold keys and values
             public double key; public double val;
             public KeyVal(double key, double val){this.key=key; this.val=val;}
             public String toString(){return "("+key+","+val+")";}
         }; 
         private Deque<KeyVal> deque;
         private int count;
         private int windowSize;


         public WindowedMaxTracker(int windowSize) {
             deque = new ArrayDeque<KeyVal>();
             count = 0;
             setWindowSize(windowSize);
         }

         public boolean isEmpty() {  return count == 0; }
         public KeyVal getMax() { return deque.getFirst(); }
         public double getMaxVal() { return deque.getFirst().val; }
         public void addPoint(double key, double val){
             if ( count>0 ) removeHead(key);
             addTail(key,val);
         }
         public void setWindowSize(int windowSize){
             this.windowSize=windowSize;
         }
         public void reset(){ deque.clear(); count=0; }
         
         /**
          * add a new point to the back of the queue, updating and removing 
          * dominated values as needed.
          */
         public void addTail(double key, double val) { addTail(new KeyVal(key,val)); }
         public void addTail(KeyVal kv) {
            // remove anything from the tail which is dominated by this new entry
            while (!deque.isEmpty() && kv.val > deque.getLast().val)
                deque.removeLast();
            // as this as the new smallest non-dominated value
            deque.addLast(kv);
            count++;
         }

         /**
          * remove an old point from the head of the queue
          */
         public void removeHead(double key, double val) { removeHead(key); }
         public void removeHead(KeyVal kv) { removeHead(kv.key); }
         public void removeHead(double key) {
            // remove value from head which is maximal
            if (count <= 0)
                throw new IllegalStateException();
            if (key-windowSize >= deque.getFirst().key){ // remove if is now outside the window
                //System.out.println("key="+key+"- ws="+windowSize+"="+(key-windowSize)+" max="+deque.getFirst().key);
                deque.removeFirst();
            }
            count--;
         }
         public String toString(){
             String str="Queue=["; for(KeyVal kv : deque) str+= kv; str+= "]";
             return str;
         }
     };


    /**
     * driver class for testing 
     */
    public static void main(String []argv){
        double data[]={1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1};
        WindowedMaxTracker mt = new WindowedMaxTracker(5);
        WindowedMaxOffsetTracker mot=new WindowedMaxOffsetTracker(5);
        for ( int i=0; i<data.length; i++){
            mt.addPoint(i,data[i]);
            mot.addPoint(i,data[i]);
            System.out.println(i + ") X="+i+" Y="+(data[i])+" diff="+(i-data[i])+" Max="+mt.getMaxVal()+" Max(diff)="+mot.b);
        }        
    }
}


/* matlab equivalent for testing
 function [this]=MaxOffsetTracker(X,Y,this)
    if(nargin<3||isempty(this)) this=struct('wght',.99,'N',-1,'X0',[],'Y0',[]'sX',0,'sY',0,'sX2',0,'sY2',0,'sXY',0); end;
    if(N<0) this.X0=X; this.Y0=Y; this.N=0; end;
		  // subtract the 0-point
		  Y  =Y - this.Y0;
		  X  =X - this.X0;
		  // update the summary statistics
		  this.N  =this.wght*this.N   + 1;
		  this.sY =this.wght*this.sY  + Y;
		  this.sX =this.wght*this.sX  + X;
		  this.sY2=this.wght*this.sY2 + Y*Y;
		  this.sYX=this.wght*this.sYX + Y*X;
		  this.sX2=this.wght*this.sX2 + X*X;
		  // update the fit parameters
		  Xvar  = this.sX2 - this.sX * this.sX / this.N;
		  YXvar = this.sYX - this.sY * this.sX / this.N;
		  if (this.N > 1.0+this.wght && Xvar>YXvar*1e-10){ // only if have good condition number (>1e-10)
				this.m = YXvar / Xvar; // NaN if origin and 1 point only due to centering
		  } else if ( N>0.0 ) { // default to gradient 1 
				this.m = 1; // TODO [] : specify a prior over the gradient
            //b = Y0 - m * X0; // TODO [] : use all the points so far...
		  }
        // update the bias given the estimated slope
        this.b = this.sY / this.N + this.Y0 - this.m * (this.sX / this.N + this.X0);
        
*/
