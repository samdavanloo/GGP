function [phi,p,tau2,sigma2,Beta,logL,invC,gamma,sStar,indcTaper] = MLE_EBLUP3(X,Y,model,no_Tries,LB,UB,p,maxEvalsSA)
% function [ phi,p,tau2,sigma2,Beta,logL,i_repeats] = MLE_EBLUP2( X,Y,model,no_Tries,LB,UB,p,maxEvalsSA)
%
% MLE_EBLUP3 finds RMLE estimates for a GP  process with noise, isotropic powered
% exponential correlation function (p fixed and not
% estimated--MLE_EBLUP_p.m does estimate p) for BIG datasets.
%The approach is based on the paper by Sang and Huang 2012 paper, titled:
%" A full scale approaximation of covariance functions for large spatial
%data sets".
%
% The code returns estimates of the parameters (phi,p,tau2,sigma2,Beta), inverse of the
% covariance matrix (invC) and the values of the parameters used in Sang
% and Huan paper (gamma,sStar,indcTaper).
%
% X must be an nx2 matrix with locations. Duplicated locations are deleted
% from X (and Y) if present since otherwise the correlation matrix is singular. We eliminate the duplicated
% locations and conduct the fitting only in the unique rows of X and Y. 
% Program also returns vector i_repeats: contains the row numbers of X that are repeated
% and that were deleted from the analysis (i.e., from X and Y)
% Y must be 1xn matrix with observed response values
% LB and UB are bounds for the values of the 3 unknowns (in this order): 
% phi(range parameter)
% sigma2 (spatial var.)
% tau2   (nugget)
% model: string. Options are:
%                   -- all of the options in x2fx
%                   --'intercept' only beta_0 is fit
%                   --'linearX1' mean model is b_1*x_1, b_1 estimated from
%                   max(X)
%                   --'linearX2' mean model is b_2*x_2, b_2 estimated from
%                   max(Y)
%                   --'linearX1int' mean is b_0 +b_1*x_1
%                   --'linearX2int' mean is b_0 +b_2*x_2
% The dimension of vector beta must match the model specified here
% no_tries: number of initial points for the optimizer, selected according
% to a latin-hypercube within the bounds.
% maxEvalsSA: no. of iterations (obj. function evaluations) for the
% simmulated annealing algorithm that does the constrained minimization
% NOTEs:       ***p is fixed in this version by the user and is now an input
%
% The paparemets that need to be tuned in this method are as foolows:
% - "m0" (preferably a square number) indirectly changes "m" which is the number of knots (m/n>0.05 is recommended by Sang and Huang 2012) 
% the relationship is m=(sqrt(m0)+1)^2 and is coming from griding that we are using for the knots.
% - "gamma" is the taper range. It is the value above which the covariance function is tapered to zero.
% - "indcTaper" is the indicator for the type of the tapering function to be used. values 1, 2 and 3 will result into 
% tapering functions spherical, Wendland1 and Wendland2, respectively.
% parameters that can be tuned are shown with right-to-left arrows (<---).

% Written by Sam D. Tajbakhsh, PSU IE dept. March, 2013.

DiffHigh=Inf;
DiffLow=0;
[n,k]=size(X);
Y=Y';


switch model
    case 'intercept'
        F=x2fx(X,zeros(1,k));
    case 'linearX1'
        x1_Only=zeros(1,k);
        x1_Only(1)=1;
        F=x2fx(X,x1_Only);
    case 'linearX1int'
        x1_Only=zeros(2,k);
        x1_Only(2,1)=1;
        F=x2fx(X,x1_Only);
    case 'linearX2'    
        x2_Only=zeros(1,k);
        x2_Only(2)=1;
        F=x2fx(X,x2_Only);
    case 'linearX2int'
        x2_Only=zeros(2,k);
        x2_Only(2,2)=1;
        F=x2fx(X,x2_Only);    
    otherwise
        F=x2fx(X,model);
end;
nopars=size(F,2); % this is the number of parameters estimated for the mean (used for REML)
% Generate initial points for the optimizer using LB and UB
% Bounds specified by user are for exp(x()), so transform them to be bounds

LB=log(LB).';UB=log(UB).';
testPoints=lhsdesign(no_Tries,3,'iterations',10000,'criterion','correlation','smooth','off');
for i=1:3
    testPoints(:,i)=testPoints(:,i)*(UB(i)-LB(i))+LB(i);
end;
xbest=LB(:)+(UB(:)-LB(:))/2;

m0=529;  %see few lines below for the exact no. of knots (m) m0=529 will result into m=576 <-----------------------------
minX=floor(min(X,[],1));
maxX=floor(max(X,[],1))+1;
gridSize=(maxX-minX)/sqrt(m0);
[X1,X2]=meshgrid(minX(1):gridSize(1):maxX(1),minX(2):gridSize(2):maxX(2));
sStar=[X1(:) X2(:)]; %locations of the knots (grid in this case)
m=size(sStar,1); % Exact No. of knots
gamma=0.01;    %taper range   <------------------------------------------------------------------------------------------
indcTaper=1; % spherical<--1  Wendland-1<--2 Wendland-2<--3   <----------------------------------------------------------


minusLogLbest=MLE3(xbest,F,Y,gamma,X,sStar,p,model,indcTaper);
for i=1:no_Tries
   xc=testPoints(i,:)';
   MLEfun = @(x) MLE3(x,F,Y,gamma,X,sStar,p,model,indcTaper);
   
   %Run SA first
   optionsSA = saoptimset('Display','iter','MaxFunEvals',maxEvalsSA);
   [xopt,minusLogL] = simulannealbnd(MLEfun,xc,LB,UB,optionsSA);
   if minusLogL<minusLogLbest
        xbest=xopt;
        minusLogLbest=minusLogL;
   end;
   display(sprintf('SA --> %5d  %8.8f   %3.4f   %2.8f   %2.8f',i,minusLogL,exp(xopt(1)),exp(xopt(2)),exp(xopt(3))));
   
   %Run fmincon next
   optionsFmincon=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-10,'DiffMaxChange',...
       DiffHigh,'DiffMinChange',DiffLow,'FinDiffType','central','MaxIter',30);
   [xopt,minusLogL]=fmincon(MLEfun,xbest,[],[],[],[],LB,UB,[],optionsFmincon);
   display(sprintf('fmincon --> %5d  %8.8f   %3.4f   %2.8f   %2.8f',i,minusLogL,exp(xopt(1)),exp(xopt(2)),exp(xopt(3))));
   if minusLogL<minusLogLbest
        xbest=xopt;
        minusLogLbest=minusLogL;
   end;
   
   display(sprintf('Best   --> %5d  %8.8f   %3.4f   %2.8f   %2.8f',i,minusLogLbest,exp(xbest(1)),exp(xbest(2)),exp(xbest(3))));
   
end; 

% Calculating parameters and logLikelihood for the Best solution
[C_l,C_nm,C_mm]=longRange(xbest,X,sStar,p); %computing the long range covariance matrix
C_s=shortRange(xbest,X,C_l,gamma,p,indcTaper); %computintg the short range covariance function
clear C_l;

mu1=eps*(10+n);  %Lophaven regularization
Temp1=C_s+(exp(xbest(3))+mu1)*speye(n);
% Temp1=C_s+(exp(x(3))+mu)*eye(n);
clear C_s;

ratio=(sum(sum(abs(Temp1)>0))/n^2);
if ratio < 0.01
    invCsTau2I=sparseinv(Temp1);
else
    Temp1=full(Temp1);
    L = chol(Temp1,'lower');
    invL=L\eye(n);
    clear L;
    invCsTau2I=invL'*invL;
%     invCsTau2I=Temp1\eye(n);
end
invCsTau2I=full(invCsTau2I);
clear Temp1;

mu2=eps*(10+m);  %Lophaven regularization
Temp2=C_mm+C_nm.'*invCsTau2I*C_nm+mu2*eye(m);
invTemp2=Temp2\eye(m);
clear Temp2;

Temp31=invCsTau2I*C_nm;
Temp32=C_nm.'*invCsTau2I;
clear C_nm;
invC=invCsTau2I-Temp31*invTemp2*Temp32;
clear invCsTau2I;
clear invTemp2;
clear Temp31 Temp32;


Y=Y';
Beta=((F'*invC*F)\eye(nopars))*F'*invC*Y;
phi=exp(xbest(1));
sigma2=exp(xbest(2));
tau2=exp(xbest(3));

logL=minusLogLbest;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MLE3 FUNCTION
function minusLogL=MLE3(x,F,Y,gamma,X,sStar,p,model,indcTaper)

[n, nopars]=size(F);
m=size(sStar,1);

[C_l,C_nm,C_mm]=longRange(x,X,sStar,p); %computing the long range covariance matrix
C_s=shortRange(x,X,C_l,gamma,p,indcTaper); %computintg the short range covariance function
clear C_l;

mu1=eps*(10+n);  %Lophaven regularization
Temp1=C_s+(exp(x(3))+mu1)*speye(n);
clear C_s;

ratio=(sum(sum(abs(Temp1)>0))/n^2);  %Sparsity Ratio 
if ratio < 0.01 %Sparsing  <-------------------------------------
    invCsTau2I=sparseinv(Temp1);
else
    Temp1=full(Temp1);
    L = chol(Temp1,'lower');
    invL=L\eye(n);
    clear L;
    invCsTau2I=invL'*invL;
end
invCsTau2I=full(invCsTau2I);
detCsTau2I1n=prod(diag(chol(Temp1)).^(2/n));
clear Temp1;

mu2=eps*(10+m);  %Lophaven regularization
Temp2=C_mm+C_nm.'*invCsTau2I*C_nm+mu2*eye(m);
invTemp2=Temp2\eye(m);
detTemp21n=prod(diag(chol(Temp2)).^(2/n));
clear Temp2;

Temp31=invCsTau2I*C_nm;
Temp32=C_nm.'*invCsTau2I;
clear C_nm;
invC=invCsTau2I-Temp31*invTemp2*Temp32;
clear invCsTau2I;
clear invTemp2;
clear Temp31 Temp32;

detCmmInv1n=(prod(diag(chol(C_mm)).^(2/n)))^(-1);
clear C_mm;
detC1n=detTemp21n*detCmmInv1n*detCsTau2I1n;

Y=Y';
FSinvF=F'*invC*F;
detFSF1n=prod(diag(chol(FSinvF)).^(2/n));

% minusLogL=log(detC1n)+(Y-F*Beta).'*invC*(Y-F*Beta)/n; %MLE with second term
% minusLogL=log(detC1n); %MLE without second term
minusLogL=log(detC1n)+log(detFSF1n);  %Harville (1974) REML 

clear invC;
minusLogL=full(minusLogL);
end

%% Long Range Covariance Matrix Builder
function [C_l, C_nm, C_mm]=longRange(x,X,sStar,p)
n=size(X,1);
m=size(sStar,1);
DD=squareform(pdist([X;sStar]));
D_nm=DD(1:n,n+1:n+m);
clear DD;
C_nm=compute_C(x,D_nm,p);
clear D_nm;
D_mm=squareform(pdist(sStar));
C_mm=compute_C(x,D_mm,p);
clear D_mm;
C_l=C_nm*(C_mm\C_nm');
end

%% Short Range Covariance Matrix Builder
function C_s=shortRange(x,X,C_l,gamma,p,indcTaper)
D_nn=squareform(pdist(X));
C_nn=compute_C(x,D_nn,p);
T=taper(D_nn,gamma,indcTaper);
clear D_nn;
T=sparse(T);  %<------------------
C_s=(C_nn-C_l).*T;
clear T;
C_s=sparse(C_s);
end

%% Taper Functions
function T=taper(D,gamma,indcTaper)

switch indcTaper
    case 1 % Spherical
        T=((max(0,1-D/gamma)).^2).*(1+D/(2*gamma));
    case 2 % Wendland-1
        T=((max(0,1-D/gamma)).^4).*(1+4*D/gamma);
    case 3 % Wendland-2
        T=((max(0,1-D/gamma)).^6).*(1+6*D/gamma+35*(D.^2)/(2*gamma^2));
end

end

%% Powered Exponential Covariance Function
function C=compute_C(x,D,p)
C=exp(x(2))*exp(-(D*exp(x(1))).^p);
end

function R=Compute_R(x,D,n,p)
% isotropic powered exponential correlation function
sigmaz2=exp(x(2))+exp(x(3));%this is a model free estimate, needed to make R a correlation matrix
R=(exp(x(2))*exp(-(D*exp(x(1))).^p)+exp(x(3))*eye(n))/sigmaz2+(10+n)*eps*eye(n); %regularization (DACE toolbox manual eq. 5.4)
end

function C_l=longRangeNew(param,X,sStar,p,Xo)
m=size(sStar,1);
nxo=size(Xo,1);
nx=size(X,1);

DDXo=squareform(pdist([Xo;sStar]));
DXo=DDXo(1:nxo,nxo+1:nxo+m);
CXo=compute_C(param,DXo,p);
D_mm=squareform(pdist(sStar));
C_mm=compute_C(param,D_mm,p);

DDX=squareform(pdist([X;sStar]));
DX=DDX(1:nx,nx+1:nx+m);
CX=compute_C(param,DX,p);

C_l=CXo*(C_mm\CX');
end

%% Short Range Covariance Matrix Builder
function C_s=shortRangeNew(param,X,C_l,gamma,p,Xo,indcTaper);
nxo=size(Xo,1);
nx=size(X,1);

DDXoX=squareform(pdist([Xo;X]));
DXoX=DDXoX(1:nxo,nxo+1:nxo+nx);
CXoX=compute_C(param,DXoX,p);
T=taper(DXoX,gamma,indcTaper);

C_s=(CXoX-C_l).*T;
end

