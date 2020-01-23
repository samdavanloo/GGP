function [X,Xo,Yp,sigma2p]=MSEPredictions(phi,p,tau2,sigma2,beta,X,Xo,Y,model)
%
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
% History
% June 5--changed the way we compute the inverse of Sigma_y, by using "\"
% instead of the very inaccurate "inv"

[n,k]=size(X);
[npred,k]=size(Xo);
% Compute Covariance matrix of the observations 
D=squareform(pdist(X));
Sigma_y=sigma2*exp(-D*phi).^p+tau2*speye(n);  %Note this include the nugget
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
c_z=zeros(1,n);
Yp=zeros(1,npred);sigma2p=zeros(1,npred);
%Compute Sigma_y inverse*(Y-F*Beta)
Sigma_yInv_YminusFbeta=Sigma_y\(Y-F*beta);
F_Sigma_yInv=F'/Sigma_y;
FSF=F_Sigma_yInv*F;
% Now compute the correlations between Z(xo) and the observations
for i=1:npred %do for each point we are predicting
    for j=1:n %for all observed points
        c_z(j)=sigma2*exp(-phi*sqrt((Xo(i,:)-X(j,:))*(Xo(i,:)-X(j,:))'))^p;
    end;
    %Now compute prediction (of Z) at point Xo(i,:)
    Yp(i)=Fo(i,:)*beta+c_z*Sigma_yInv_YminusFbeta;
    %MSE prediction error
    cSigmaInv=c_z/Sigma_y;
    F0_FSc=Fo(i,:)-(F_Sigma_yInv*c_z')';
    sigma2p(i)=sigma2-cSigmaInv*c_z'+F0_FSc/FSF*F0_FSc';
end
