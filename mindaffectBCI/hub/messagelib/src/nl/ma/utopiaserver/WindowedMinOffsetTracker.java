/*
 * Copyright (C) 2014, Jason Farquhar
 * Donders Institute for Donders Institute for Brain, Cognition and Behaviour,
 * Centre for Cognition, Radboud University Nijmegen,
 */
package nl.ma.utopiaserver; // test
import java.util.Deque;
import java.util.ArrayDeque;

public class WindowedMinOffsetTracker implements ClockAlignment {
	 public double m, b;         // fit, scale and offset
	 double sErr=0;              // running estimate of the est-true Y error
    double alpha=.999;
    WindowedMinTracker windowedMin;
    
	 public WindowedMinOffsetTracker(){ this(60*1000); } // default to 1min(=60000ms) window size
    public WindowedMinOffsetTracker(int windowSize){
        windowedMin = new WindowedMinTracker(windowSize);
        reset();
	 }
	 public void setHalfLife(int hl){
        windowedMin.setWindowSize(hl);
	 }

	 public void reset(){
        windowedMin.reset();
		  m=1; b=0;
	 }

	 public void addPoint(double X, double Y) {
        double diff=Y-X;
        //System.out.println("X="+X+"diff="+diff);
        windowedMin.addPoint(X,diff);
        double ob=b;
        b = windowedMin.getMinVal();
        if( ob!=b ){
            System.out.println("\nOffset Update:" + ob + "->" + b );
        }
        double estErr = Math.abs(Y - getY(X));
        sErr = alpha*sErr + (1.0-alpha)*estErr; // moving average error estimate
	 }

	 public double getX(double Y){  return (Y-b)/m; }
	 public double getY(double X){  return m*X + b; }	 
	 public double getYErr(){       return sErr; }

    public String toString(){
        return "Y= " + m + "*X + " + b + " (+/- " + getYErr() + ") ";
    }


     static final class WindowedMinTracker {
         // deque to hold the set of non-dominated values in this window.
         // A value is non-dominated if there are no smaller values
         // in front of it in the queue. (i.e. if the queue remains in sorted order)
         // Thus; the 1st entry in the queue is the largest value in the window
         // and the 2nd entry next largest *with only smaller values before it* etc.
         public static final class KeyVal {// struct to hold keys and values
             public double key; public double val;
             public KeyVal(double key, double val){this.key=key; this.val=val;}
             public String toString(){return "("+key+","+val+")";}
         }; 
         private Deque<KeyVal> deque;
         private int count;
         private int windowSize;


         public WindowedMinTracker(int windowSize) {
             deque = new ArrayDeque<KeyVal>();
             count = 0;
             setWindowSize(windowSize);
         }

         public boolean isEmpty() {  return count == 0; }
         public KeyVal getMin() { return deque.getFirst(); }
         public double getMinVal() { return deque.getFirst().val; }
         public void addPoint(double key, double val){
             if ( count>0 ) removeHead(key);
             addTail(key,val);
         }
         public void setWindowSize(int windowSize){
             this.windowSize=windowSize;
         }
         public void reset(){ deque.clear(); count=0; }
         
         public void addTail(double key, double val) { addTail(new KeyVal(key,val)); }
         public void addTail(KeyVal kv) {
            // remove anything from the tail which is dominated by this new entry
            while (!deque.isEmpty() && kv.val < deque.getLast().val)
                deque.removeLast();
            // as this as the new smallest non-dominated value
            deque.addLast(kv);
            count++;
         }
         
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


    // driver for testing
    public static void main(String []argv){
        double data[]={1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1};
        WindowedMinTracker mt = new WindowedMinTracker(5);
        WindowedMinOffsetTracker mot=new WindowedMinOffsetTracker(5);
        for ( int i=0; i<data.length; i++){
            mt.addPoint(i,data[i]);
            mot.addPoint(i,data[i]);
            System.out.println(i + ") X="+i+" Y="+(data[i])+" diff="+(i-data[i])+" Min="+mt.getMinVal()+" Min(diff)="+mot.b);
        }        
    }
}
