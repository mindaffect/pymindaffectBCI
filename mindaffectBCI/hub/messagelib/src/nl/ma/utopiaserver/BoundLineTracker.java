/*
 * Copyright (C) 2019, Jason Farquhar
 * MindAffect B.V.
 */
package nl.ma.utopiaserver; // test

/**
 * Windowed running maximum tracking from an input stream of values.
 */
public class BoundLineTracker implements ClockAlignment {
	 public double m, b;         // fit, scale and offset
    double[] xs, ys;
    double Cpm=50;
    double halflife=0;
    boolean minweightreplacement=false;
    boolean lowerboundp=true;
    double tol=1e-4;
    int N=0;
    int minIter=1;
    int maxIter=30;
    
	 public BoundLineTracker(){
        this(-5);
    } 
    public BoundLineTracker(int windowSize){this(windowSize,true,-1);}
    public BoundLineTracker(int windowSize, boolean lowerboundp, double halflife){
        this(windowSize,lowerboundp,halflife,50);
    }
    public BoundLineTracker(int windowSize, boolean lowerboundp, double halflife, double Cpm){
        N=-windowSize;
        xs=new double[windowSize];
        ys=new double[windowSize];
        this.lowerboundp=lowerboundp;
        this.halflife=halflife;
        if( halflife<0 ) {
            minweightreplacement=true;
            halflife=-halflife;
        }
        this.Cpm=Cpm;
        if( Cpm < 0 ) {
            System.err.println("Negative cost!!!");
            Cpm = -Cpm;
        }
	 }

    /**
     * reset the regression information to initial (memory free) state
     */
	 public void reset(){
		  m=1; b=0; N=-xs.length;
	 }

    /**
     * add a new point to the regression.
     * @param X - the new points X value
     * @param Y - the new points Y value
     */
	 public void addPoint(double X, double Y) {
        double[] xs=null, ys=null;
        //System.out.println("X="+X+" Y="+Y+ "   N="+N+" m="+m+" b="+b);
        if( N<0 ){ // warmup
            // insert at the end
            int idx=this.xs.length+N; // fill from the front
            this.xs[idx]=X;
            this.ys[idx]=Y;
            // copy valid to new array for simplicity
            xs=new double[idx+1]; System.arraycopy(this.xs,0,xs,0,xs.length);
            ys=new double[idx+1]; System.arraycopy(this.ys,0,ys,0,ys.length);
        } else {
            int idx = N;
            this.xs[idx]=X;  xs=this.xs;
            this.ys[idx]=Y;  ys=this.ys;
        }
        N=N+1; if( N>=xs.length ) N=0;

        // don't fit if not enough points
        if( xs.length < 2 ) {
            b=Y-X;
            return;
        }
        if( xs.length<3 ) {
            if( xs[1]-xs[0] > 0 && ys[1]-ys[0]>0 ){
                m = (xs[1]-xs[0])/(ys[1]-ys[0]);
            } else {
                m = 1;
            }
            b = (ys[0]+ys[1])/2.0f - m*(xs[0]+xs[1])/2.0f;
            return;
        }
        
        // compute xs 1,2nd cummulants
        // center xs to reduce numerical issues
        double sX=0, sX2=0, maxx=xs[0];
        for ( int i=0; i<xs.length; i++){
            sX=sX+xs[i];
            sX2=sX2+xs[i]*xs[i];
            if( xs[i]>maxx ) maxx=xs[i];
        }
        double muX=sX/(xs.length);
        double sigmaX=Math.sqrt(sX2/xs.length-muX*muX);
        //float muX=sX/(xs.length);
        //for ( int i=0; i<xs.length; i++){
        //    xs[i] = xs[i]-muX;
        //}
        double c=sigmaX;
        //System.out.println("muX="+muX+" sigmaX="+sigmaX+" c="+c);
        b=b/c;

        double J=9999999999.0d;
        double oJ=J;
        double om=m;
        double ob=b;
        double[] w= new double[xs.length];
        double[] yest=new double[xs.length];
        double[] yerr=new double[xs.length];
        double xwc=0, xwx=0, xwy=0, ywy=0, ywc=0, cwc=0;
        for ( int iter=1; iter<maxIter; iter++){
            // state for convergence testing
            om=m; ob=b; oJ=J;
            //System.out.println(iter + ") y = " + m + "*x + " + b + "*" + c );
            // compute the current estimate, and point weighting
            xwc=0; xwx=0; xwy=0; ywy=0; ywc=0; cwc=0; J=0;
            for ( int i=0; i<xs.length; i++){
                yest[i] = xs[i]*m + b*c;
                yerr[i] = yest[i] - ys[i];
                // initialize the point weight
                if( halflife > 0 ) {
                    w[i]=Math.pow(2,-(maxx-xs[i])/halflife);
                } else {
                    w[i]=1.0d;
                }
                // lowerbound->  soft-loss (l1) when y>yest => yerr=yest-y<0
                if( lowerboundp && yerr[i]<0 ) {
                    double ow=w[i];
                    w[i]= w[i]/Cpm/((-yerr[i])+tol); // regularize for numerics
                    
                // upperbound ->  soft-loss (l1) when yest<y => yerr=yest-y>0 
                }else if ( !lowerboundp && yerr[i]>0 ) { // upper-bound, est is higher, i.e. yest>y
                    
                    // Danger: careful of the sign here!
                    w[i]= w[i]/Cpm/(yerr[i]+tol);
                }
                // update the object value
                J   = J + w[i]*yerr[i]*yerr[i];
                // accumulate summary statistics
                xwc = xwc + xs[i]*w[i]*c;
                xwx = xwx + xs[i]*w[i]*xs[i];
                xwy = xwy + xs[i]*w[i]*ys[i];
                ywy = ywy + ys[i]*w[i]*ys[i]; // not used
                ywc = ywc + ys[i]*w[i]*c;
                cwc = cwc + c*w[i]*c;
                //System.out.println("pt" + i + " x="+xs[i]+" y="+ys[i] + "  y_est="+yest[i]+" yerr="+yerr[i]+" w="+w[i]);
            }
            // solve the weighted least squares
            // using the 2x2 matrix inverse formula:
            // C^-1 =   [ xwx xwc ]^-1 = [cwc  -xwc] * 1/det(C)
            //          [ xwc cwc ]      [-xwc  xwx]
            // where det(C) = xwx*cwc - xwc*xwc
            // Hence:
            //  C^-1*[xwy] = 1/det(c) * [  cwc*xwy - xwc*ywc ]
            //       [ywc]              [ -xwc*xwy + xwx*ywc ]
            double detC = xwx*cwc-xwc*xwc;
            if( detC<1e-5 ) {
                System.out.println(iter+") Warning: near singular matrix, detC="+detC);
                System.out.println("[xwx xwc] = " + xwx + " " + xwc);
                System.out.println("[xwc cwc] = " + xwc + " " + cwc);
                // add a ridge to try and stabilize
                //double ridge=(xwx+cwc)*1e-3; // scaled by diag
                //detC= detC+ridge;
                //xwx = xwx+Math.sqrt(ridge);
                //cwc = cwc+Math.sqrt(ridge);
            }
            m = ( cwc*xwy - xwc*ywc)/detC;
            b = (-xwc*xwy + xwx*ywc)/detC;
            // validate the solution
            //double m1 = xwx*m + xwc*b;
            //double m2 = xwc*m + cwc*b;
            //System.out.println("M1: xwx*m+xwc*b="+m1+" xwy="+xwy);
            //System.out.println("M2: xwc*m+cwc*b="+m2+" ywc="+ywc); 

            
            // convergence test
            double normab      = m*m           + b*b;
            double normdeltaab = (om-m)*(om-m) + (ob-b)*(ob-b);
            //System.out.println(iter+") J="+J+ " dab="+normdeltaab);
            if ( iter>=minIter ) {
                if( Math.abs(oJ-J)/(J+1e-8)<tol
                    || normdeltaab/normab<tol ){
                    //System.out.println(iter + " iter");
                    break;
                }
            }
        }
        // unto the c-scaling for b
        b = b * c;
        
        // find the min-weight point to be replaced next time
        if( minweightreplacement && N>=0 ){
            double minval=w[0];
            int minidx=0;
            for ( int i=1; i<w.length; i++){
                if( w[i]<minval ){
                    minval=w[i]; minidx=i;
                }
            }
            N=minidx;
        }
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
	 public double getYErr(){       return 1000; }

    public String toString(){
        return "Y= " + m + "*X + " + b + " (+/- " + getYErr() + ") ";
    }


    public static void testcases(){
        {
        //double[] noise={0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1};
        double[] noise={0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1};
        // data is linear unit-gain trend + noise
        double[] data = new double[noise.length];
        for ( int i=0; i<data.length; i++) { data[i]=i+noise[i]; }
        // run this test
        BoundLineTracker mt = new BoundLineTracker(5,false,10);
        double Yest=0;
        for ( int i=0; i<data.length; i++){
            mt.addPoint(i,data[i]);
            Yest = mt.getY(i);
            System.out.println(i + ") X (+Ytrue)="+i+" Y="+(data[i])+" Yest=" + Yest + " diff="+(Yest-i));
            System.out.println(i+") "+mt);
        }
        }

        {
            double[] xs={1,2,3,4,5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
            double[] ys={5,5,5,5,5,10,10,10,10,10,15,15,15,15,15,20,20,20,20,20,25,25,25,25,25,30,30,30,30,30};
            for ( int i=0; i<ys.length; i++) { ys[i]=ys[i]+100000; }
        // run this test
        BoundLineTracker mt = new BoundLineTracker(500,true,-1,100);
        double Yest=0;
        for ( int i=0; i<xs.length; i++){
            mt.addPoint(xs[i],ys[i]);
            Yest = mt.getY(xs[i]);
            System.out.println(i + ") X (+Ytrue)="+xs[i]+" Y="+(ys[i])+" Yest=" + Yest + " diff="+(Yest-ys[i]));
            System.out.println(i+") "+mt);
        }
        }
    }
    
    /**
     * driver class for testing 
     */
    public static void main(String []argv){
        System.out.println("Usage: BoundLineTracker filename N lowerboundp halflife cpm");
        java.io.BufferedReader bfr=null;
        //System.out.println("argv["+argv.length+"]"+argv);
        if( argv.length==0 ) {
            testcases();
            return;
        }
        String filename=null;
        if ( argv.length>0 ){
            filename=argv[0];
        }
        System.out.println("filename="+filename);
        int N=500;
        if( argv.length>1 ){
            try{
            N = Integer.valueOf(argv[1]);
            } catch ( NumberFormatException e) {
            }
        }
        System.out.println("N="+N);
        boolean lowerboundp=true;
        if( argv.length>2 ){
            try{
                lowerboundp = Boolean.valueOf(argv[2]);
            } catch ( NumberFormatException e) {
            }
        }
        System.out.println("lowerboundp="+lowerboundp);
        float halflife=10*1000;
        if( argv.length>3 ){
            try{
                halflife = Float.valueOf(argv[3]);
            } catch ( NumberFormatException e) {
            }
        }
        System.out.println("halflife="+halflife);
        float cpm=10;
        if( argv.length>4 ){
            try{
                cpm = Float.valueOf(argv[4]);
            } catch ( NumberFormatException e) {
            }
        }
        System.out.println("Cpm="+cpm);

        // read the test data from stdin/file
        if( filename.equals("-") ) { // read from std-in
            bfr = new java.io.BufferedReader(new java.io.InputStreamReader(System.in));
        } else { // read from filename
            try{
            bfr = new java.io.BufferedReader(new java.io.FileReader(filename));
            } catch ( java.io.IOException ex ) {
                ex.printStackTrace();
                System.exit(-1);
            }
        }
        String line;
        java.util.ArrayList<Float> xs=new java.util.ArrayList<Float>();
        java.util.ArrayList<Float> ys=new java.util.ArrayList<Float>();
        try { 
            while( (line = bfr.readLine())!=null){
                if ( line == null || line.startsWith("#") || line.length()==0 )
                    continue;
                String[] values=line.trim().split("[, \t]");
                xs.add(Float.valueOf(values[0]));
                ys.add(Float.valueOf(values[1]));            
            }
        } catch ( java.io.IOException ex ) {
            ex.printStackTrace();
            System.out.println("error reading test matrix");
        }


        BoundLineTracker mt = new BoundLineTracker(N,lowerboundp,halflife,cpm);
        double Yest=0,x,y;
        for ( int i=0; i<xs.size(); i++){
            x=xs.get(i);
            y=ys.get(i);
            mt.addPoint(x,y);
            Yest = mt.getY(x);
            System.out.println(i + ") X (+Ytrue)="+x+" Y="+(y)+" Yest=" + Yest + " diff="+(Yest-y));
            System.out.println(i+") "+mt);
        } 
            
    }
}
