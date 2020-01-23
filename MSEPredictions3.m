function [X,Xo,Yp,sigma2p]=MSEPredictions3(phi,p,tau2,sigma2,beta,X,Xo,Y,model,SigmaYinv,gamma,sStar,indcTaper)

%Computes MMSE predictors of a Gaussian Process with a powered exponential
%correlation function and mean given by a linear model f(x)'*beta
%Also returns the estimated MSE of the predictions using the plug-in
%estimator of the variance of the prediction
%Assumes locations are 2-dimensional, ans assumes model is
%
%  Y(x) = Z(x) + e(x), e(x)~iidN(0,tau2) and
%  Z(x) = N(f(x)'beta, C_z(0)) and cov(Z(x),Z(x'))=C_z(x,x')
%
% where C_z(x,x')=sigma2*exp(-phi*|x-x'|)^p  (note there is NO nugget in
% the "state", i.e., the state is smooth).
% 
% The goal is to predict Z(x), not Y(x). The data are the observations 
% (Y(x_1),...,Y(x_n)). 

% X =n locations where the model was fit (nx3)
% Xo=npred locations where we want to compute the MMSE predictions (npred x 3)
% Y=n observed responses (nx1)
% model: string. Options are:
%                   --all of the options in x2fx
%                   --'intercept' only beta_0 is fit
%                   --'linearX1' mean model is a constant equal to b*x_1
%                   --'linearX2' mean model is a constant equal to b*x_2
%                   --'linearX1int' mean is b_0 +b_1*x_1
%                   --'linearX2int' mean is b_0 +b_2*x_2
% The dimension of vector beta must match the model specified here
%
% Written by Sam D. Tajbakhsh, PSU IE dept. March, 2013.

[nx,k]=size(X);
[nxo,k]=size(Xo);

switch model
    case 'intercept'
        F=x2fx(X,zeros(1,k));
        Fo=x2fx(Xo,zeros(1,k));
    case 'linearX1'
        x1_Only=zeros(1,k);
        x1_Only(1)=1;
        F=x2fx(X,x1_Only);
        Fo=x2fx(Xo,x1_Only);
    case 'linearX1int'
        x1_Only=zeros(2,k);
        x1_Only(2,1)=1;
        F=x2fx(X,x1_Only);
        Fo=x2fx(Xo,x1_Only);
    case 'linearX2'    
        x2_Only=zeros(1,k);
        x2_Only(2)=1;
        F=x2fx(X,x2_Only);
        Fo=x2fx(Xo,x2_Only);
    case 'linearX2int'
        x2_Only=zeros(2,k);
        x2_Only(2,2)=1;
        F=x2fx(X,x2_Only);      
        Fo=x2fx(Xo,x2_Only);
    otherwise
        F=x2fx(X,model);
        Fo=x2fx(Xo,model);
end;

param=log([phi,sigma2,tau2]);
C_l=longRangeNew(param,X,sStar,p,Xo);
C_s=shortRangeNew(param,X,C_l,gamma,p,Xo,indcTaper);
C_z=C_l+C_s;

Yp=Fo*beta+C_z*SigmaYinv*(Y-F*beta);
sigma2p=sigma2*ones(nxo,1)-diag(C_z*SigmaYinv*C_z')+tau2*ones(nxo,1);

end

%% Long Range Covariance Matrix Builder
function C_l=longRangeNew(param,X,sStar,p,Xo)
m=size(sStar,1);
nxo=size(Xo,1);
nx=size(X,1);

DDXo=squareform(pdist([Xo;sStar]));
DXo=DDXo(1:nxo,nxo+1:nxo+m);
CXo=compute_C(param,DXo,p);
clear DXo;
D_mm=squareform(pdist(sStar));
C_mm=compute_C(param,D_mm,p);
clear D_mm;

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
clear DXoX;

C_s=(CXoX-C_l).*T;
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

function C=compute_C(param,D,p)
C=exp(param(2))*exp(-(D*exp(param(1))).^p);
end