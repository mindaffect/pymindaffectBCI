/*
 * Copyright (C) 2014, Jason Farquhar
 * Donders Institute for Donders Institute for Brain, Cognition and Behaviour,
 * Centre for Cognition, Radboud University Nijmegen,
 */
package nl.ma.utopiaserver;

/** 
 * Perform a running expionentially weighted regression, with outlier suppression
 */
public class ExpWeightedRegression implements ClockAlignment {
	 double Y0, X0; // starting samples, X
	 public double Xlast=-1000, Ylast=-1000; // last X at which the sync was updated, needed to compute accuracy?
	 double N=-1;  // number points
	 double sY=0, sX=0;          // sum samples, X
	 double sY2=0, sYX=0, sX2=0; // sum product samples X
	 public double m, b;         // fit, scale and offset
	 double alpha, hl;           // learning rate, halflife
	 double sErr=0;              // running estimate of the est-true Y error
	 double minUpdateX = 100;    // only update if at least 100ms apart, prevent rounding errors
	 double totalWeight = 0 ;    // total summed weight with infinite time
    double warmupWeight = 0;    // amount of weight before we say out of warmup

	 // Note to make this work reliabily we use a combination of a long averagering interval
	 // AND a rapid outlier detection to rapidly detect systematic changes which require 
	 // discarding the memory

    public ExpWeightedRegression() {this(.999999);}
    /**
     * Construct expionentially weighted regression class.
     * @param alpha -- decay constant for the exponential weighting
	  *  N.B. half-life = log(.5)/log(alpha) 
	  *     alpha     = exp(log(.5)/half-life) .8=3, .9=7, .95=13, .97=22, .98=34, .99=69 updates
	  * Summed updates = 1/(1-alpha)
     */
	 public ExpWeightedRegression(double alpha){
	 	setAlpha(alpha);
	 	reset();
	 }
    /**
     * Construct exponentially weighted regression, and set seed points.
     * @param Y - initial y value
     * @param X - initial x value
     * @param alpha -- decay constant for the weighting.
     */
	 public ExpWeightedRegression(double Y, double X, double alpha) {
		  this(alpha);
		  addPoint(X,Y);
	 }

    /**
     * set the minimum X difference for a point to be treated as a new point.
     */
	 public int setMinUpdateX(int minUpdateX){ return (int)(this.minUpdateX=minUpdateX); }

    /**
     * set the decay constant for the exponential weighting.
     * @param alpha -- the decay constant
     */
	 public void setAlpha(double alpha){
	 // N.B. the max weight is: \sum alpha.^(i) = 1/(1-alpha)
	 //      and the weight of 1 half-lifes worth of data is : (1-alpha.^(hl))/(1-alpha);
		 this.alpha=alpha;
		 this.hl = Math.log(.5)/Math.log(alpha);
		 this.totalWeight  = 1/(1-this.alpha);
       this.warmupWeight = (1-Math.pow(this.alpha,3))/(1-this.alpha);
	 }

    /**
     * set the decay constant for the weighting, from a given half-life.
     * @param hl - the half life to use
     */
	 public void setHalfLife(double hl){
	 	double alpha = Math.exp(Math.log(.5)/hl);
	 	setAlpha(alpha);
	 }

    /**
     * reset the regression information to initial (memory free) state
     */
	 public void reset(){
		  System.out.println("reset clock");
		  N=0;
		  Y0=0; X0=0;
        sY=0; sX=0;
        sY2=0; sYX=0; sX2=0;
        Xlast=0; Ylast=0;
        sErr=1;
		  m=1; b=0;
	 }

    /**
     * add a new point to the regression.
     * @param X - the new points X value
     * @param Y - the new points Y value
     */
	 public void addPoint(double X, double Y) {
		  if ( N <= 0 ){ // first call with actual data, record the start points
				N=1; Y0=Y; X0=X; Xlast=X; Ylast=Y; b=Y0-X0; m=1; sErr=1;
            return;
		  } else if ( Y==Ylast || X==Xlast ) {
				//System.out.println("Too soon! Y=" + Y + " Ylast=" + Ylast + " X=" + X + " Xlast=" + Xlast);
				// sanity check inputs and ignore if too close in time or Y number 
				// -> would lead to infinite gradients
				return;
		  }
		  // Update the Y error statistics
		  double estErr = Math.abs(getY(X)-Y);

        // outlier rejection
        if( N>warmupWeight &&
            estErr > 2f*sErr/N*(X-X0)/(Xlast-X0+1) ) {
            sErr = sErr*1.1; // increase the error window, so eventually points will be accepted
            //System.out.println("Outlier!");
            return;
        }
        sErr = sErr * alpha + estErr; // moving average error estimate

		  // BODGE: this should really the be integerated weight
        // weight based on time since last update
		  double wght = alpha; // Math.pow(alpha,((double)(X-Xlast))/1000.0); 
		  Xlast=X;
        Ylast=Y;
		  // subtract the 0-point
		  Y  =Y - Y0;
		  X  =X - X0;
        //System.out.println("Updating...");
		  // update the summary statistics
		  N  =wght*N   + 1;
		  sY =wght*sY  + Y;
		  sX =wght*sX  + X;
		  sY2=wght*sY2 + Y*Y;
		  sYX=wght*sYX + Y*X;
		  sX2=wght*sX2 + X*X;

		  // update the fit parameters
		  double Yvar  = sY2 - sY * sY / N;
		  double Xvar  = sX2 - sX * sX / N;
		  double YXvar = sYX - sY * sX / N;
        // for any geometric series: s(n)=\sum_1:n a*1 + r*s(n-1)= a + a*r + ... + a*r^{n-1} = a*(1-r^n)/(1-r)
        // Thus, when a=1, and n=inf = 1/(1-r)
        // Thus, warmup fin when at least 3 points: (1-wght.^3)./(1-wght)
		  if ( N > warmupWeight ){ // only if have good condition number (>1e-10)
			  m = (YXvar / Xvar + Yvar/YXvar)/2; // NaN if origin and 1 point only due to centering
		  } else if ( N>0.0 ) { // default max range gradient
			  m = (Ylast - Y0) / (Xlast - X0);
		  } else {
		  	  m=1.0;
		  }
        // update the bias given the estimated slope
        b = (Y0 + sY / N) - m * (X0 + sX / N);
		  //System.out.println();
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
	 public double getYErr(){
        if( N > warmupWeight )
            return sErr/N;
        else
            return 10000;
	 }
    
    public String toString(){
        return "Y= " + m + "*X + " + b + " (+/- " + getYErr() + ") " + "N="+N;
    }
}


/* matlab equivalent for testing
 function [this]=ExpWeightedRegression(X,Y,this)
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
