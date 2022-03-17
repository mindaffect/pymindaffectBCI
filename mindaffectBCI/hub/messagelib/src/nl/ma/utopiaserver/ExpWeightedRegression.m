function [this]=ExpWeightedRegression(X,Y,this)
  if(nargin<3||isempty(this))
    this=struct('alpha',.99,'N',-1,'X0',[],'Y0',[],'Xlast',[],'Ylast',[],'sX',0,'sY',0,'sX2',0,'sY2',0,'sYX',0,'m',[],'b',[],'sErr',0,'totalWeight',[]);
    this.totalWeight = 1./(1-this.alpha);
    this.warmupWeight= (1-this.alpha.^3)./(1-this.alpha); % >3 points for warmup
    return;
  end;
  if(this.N<0)
    this.X0=X; this.Y0=Y; this.N=0; this.b=this.Y0-this.X0; this.m=1; this.sErr=0;
    return;
  elseif( Y==this.Ylast || X==this.Xlast )
    return;
  end;

                                % running average estimate error
  estErr = abs(Y - (X*this.m + this.b));
  % outlier rejection regression... reject points too far from the average error
  if( this.N>this.warmupWeight && estErr > 4*(this.sErr/this.N)*(X-this.X0)/(this.Xlast-this.X0) )
    this.sErr = this.sErr * 1.1; % inc the error estimate
    return;
  end; 
  this.sErr = this.sErr * this.alpha + estErr;

  this.Xlast=X;
  this.Ylast=Y;
  
  %% subtract the 0-point
  Y  =Y - this.Y0;
  X  =X - this.X0;
  %% update the summary statistics
  wght    = this.alpha;
  this.N  = wght*this.N   + 1;
  this.sY = wght*this.sY  + Y;
  this.sX = wght*this.sX  + X;
  this.sY2= wght*this.sY2 + Y*Y;
  this.sYX= wght*this.sYX + Y*X;
  this.sX2= wght*this.sX2 + X*X;
  %% update the fit parameters
  Yvar  = this.sY2 - this.sY * this.sY / this.N;
  Xvar  = this.sX2 - this.sX * this.sX / this.N;
  YXvar = this.sYX - this.sY * this.sX / this.N;
  if (this.N > this.warmupWeight ) %% only if have enough points
    this.m = (YXvar / Xvar + Yvar/YXvar )/2; %% NaN if origin and 1 point only due to centering
  elseif ( this.N>0.0 ) %% default to gradient 1 
	 this.m = (this.Ylast-this.Y0)/(this.Xlast-this.X0);
  else
    this.m=1;
  end
  
  %% update the bias given the estimated slope
  this.b = (this.Y0 + this.sY / this.N ) - this.m * (this.X0 + this.sX / this.N);
  return;   
   
  function b=testcase()
    state=ExpWeightedRegression();
    e=[];
    Y2=[];Y2(1)=Y(1);
    for i=1:size(X,2)-1;
      state=ExpWeightedRegression(X(i),Y(i),state);
      Y2(i+1) = state.m*X(i+1)+state.b;
      e(i) = state.sErr./state.N;
    end
    clf;plot(X,[Y;Y2]');legend('Y','Yest');
    clf;plot(X'-[Y;Y2]'-X(1)-Y(1));
    clf;subplot(211);plot(X'-[Y;Y2]'-X(1)-Y(1));subplot(212); plot(e,'r'); legend('Y','Yest','err')
