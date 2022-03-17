/*
 * Copyright (C) 2014, Jason Farquhar
 * Donders Institute for Donders Institute for Brain, Cognition and Behaviour,
 * Centre for Cognition, Radboud University Nijmegen,
 */
package nl.ma.utopiaserver;

public class MinOffsetTracker implements ClockAlignment {
	 public double Xlast=-1000, Ylast=-1000; // last X at which the sync was updated, needed to compute accuracy?
	 double N=-1;  // number points
	 public double m, b;         // fit, scale and offset
	 double alpha;           // learning rate, halflife
	 double sErr=0;              // running estimate of the est-true Y error

	 // Note to make this work reliabily we use a combination of a long averagering interval
	 // AND a rapid outlier detection to rapidly detect systematic changes which require 
	 // discarding the memory
	 //N.B. half-life = log(.5)/log(alpha) 
	 //     alpha     = exp(log(.5)/half-life) .8=3, .9=7, .95=13, .97=22, .98=34, .99=69 updates
	 // Summed updates = 1/(1-alpha)
	 public MinOffsetTracker() {this(.999999);}
	 public MinOffsetTracker(double alpha){
	 	setAlpha(alpha);
	 	reset();
	 }
	 public MinOffsetTracker(double Y, double X, double alpha) {
		  this(alpha);
		  addPoint(X,Y);
	 }

	 public void setAlpha(double alpha){
	 // N.B. the max weight is: \sum alpha.^(i) = 1/(1-alpha)
	 //      and the weight of 1 half-lifes worth of data is : (1-alpha.^(hl))/(1-alpha);
		 this.alpha=alpha;
	 }

	 public void setHalfLife(double hl){
	 	double alpha = Math.exp(Math.log(.5)/hl);
	 	setAlpha(alpha);
	 }

	 public void reset(){
		  N=0;
        Xlast=0; Ylast=0;
        sErr=1;
		  m=1; b=0;
	 }

	 public void addPoint(double X, double Y) {
		  if ( N <= 0 ){ // first call with actual data, record the start points
				N=1; Xlast=X; Ylast=Y; b=Y-X; m=1; sErr=1;
            return;
		  }
        // get the new offset between the 2 and update if it's smaller
        // => X is as big as possible for a given Y
        double diff = Y-X;
        if( diff < b ) {
            if (N < 100) { // BODGE: stop tracking after sufficient number calls..
				System.out.println(N + "New offset " + b + "->" + diff);
                b = diff;
            }
        }
        double estErr = Math.abs(Y - getY(X));
        N    = alpha*N    + 1;
        sErr = sErr*alpha + estErr; // moving average error estimate
	 }

	 public double getX(double Y){  return (Y-b)/m; }
	 public double getY(double X){  return m*X + b; }	 
	 public double getYErr(){       return sErr/N; }

    public String toString(){
        return "Y= " + m + "*X + " + b + " (+/- " + getYErr() + ") " + "N="+N;
    }
}


/* matlab equivalent for testing
 function [this]=MinOffsetTracker(X,Y,this)
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
