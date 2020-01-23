function [ phi,p,tau2,sigma2,Beta,logL] = MLE_EBLUP2( X,Y,model,no_Tries,LB,UB,p,maxEvalsSA)
% function [ phi,p,tau2,sigma2,Beta,logL] = MLE_EBLUP2( X,Y,model,no_Tries,LB,UB,p,maxEvalsSA)
%
% MLE_EBLUP2 finds RMLE estimates for a GP  process with noise, isotropic powered
% exponential correlation function (p fixed and not estimated--MLE_EBLUP_p.m does estimate p). 
% X must be an nx2 matrix with locations. Duplicated locations are deleted
% from X (and Y) if present since otherwise the correlation matrix is singular. We eliminate the duplicated
% locations and conduct the fitting only in the unique rows of X and Y. 
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
% Written by EDC, PSU IE dept. May 23, 2012
% History
% June 7, 2013: revert to earlier version where matrix D is computed here; this
% is OK for small datasets. We still clear matrices that are not used and
% used speye instead of eye.
% June 7, 2012: Add a final step of fmincon at the best point found by SA.
% Standardize all vars before estimation, and un-standardize at the
% end to display and compute errors.  
% June 5, 2012: changed MLE for REML, following the Santner et al. book (p. 67), after correction
% as the book has an error on that page.
% June 4, 2012: new parameter maxEvalsSA: the # of function evaluations in
% a single simulated annealing run
% June 1 2012: defned new models: 'linearX1int' and 'linearX2int' which
% estimate intercept and the slope of x or y only.
% May 30, 2012: Set the non-negative variables tau2,sigma2 and phi equal to exp(alpha_i)
% following the Xia, Ding, Wang 2008 IIE Trans. paper. The "DACE" toolbox
% (Lophaven et al. 2002) MLE formulation is used. 


DiffHigh=Inf;
DiffLow=0;
[n,k]=size(X);
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
D=squareform(pdist(X));
nopars=size(F,2); % this is the number of parameters estimated for the mean (used for REML)
% Generate initial points for the optimizer using LB and UB
% Bounds specified by user are for exp(x()), so transform them to be bounds
% for x()
LB=log(LB);UB=log(UB);
testPoints=lhsdesign(no_Tries,3,'iterations',10000,'criterion','correlation','smooth','off');
for i=1:3
    testPoints(:,i)=testPoints(:,i)*(UB(i)-LB(i))+LB(i);
end;
xbest=LB(:)+(UB(:)-LB(:))/2;
xbest=xbest';
Lbest=MLE2(xbest,n,F,Y,p,model,nopars);
for i=1:no_Tries
   xc=testPoints(i,:)';
   [xopt,L] = runnestedSA(xc,n,F,Y,p,model,LB,UB,maxEvalsSA,nopars);
   display(sprintf('SA     --> %5d  %8.8f   %3.4f   %2.8f   %2.8f',i,L,exp(xopt(1)),exp(xopt(2)),exp(xopt(3))));
   [xopt,L, flag]=fmincon(@MLE2,xopt,[],[],[],[],LB,UB,[],optimset('Algorithm','interior-point','Display','off','TolFun',1e-10,'DiffMaxChange',DiffHigh,'DiffMinChange',DiffLow,'FinDiffType','central'),n,F,Y,p,model,nopars);
   display(sprintf('fmincon-->%5d  %8.8f   %3.4f   %2.8f   %2.8f',i,L,exp(xopt(1)),exp(xopt(2)),exp(xopt(3))));
   if L<Lbest
        xbest=xopt;
        Lbest=L;
   end;
   display(sprintf('Best   --> %5d  %8.8f   %3.4f   %2.8f   %2.8f',i,Lbest,exp(xbest(1)),exp(xbest(2)),exp(xbest(3))));
end; 
R=Compute_R(xbest,n,p);
C=chol(R,'lower'); 
Cinv=C\speye(n);
clear C;
%         Ytilde=C\Y';
%         Ftilde=C\F;
Ytilde=Cinv*Y;
Ftilde=Cinv*F;
clear Cinv;
[Q,Gt]=qr(Ftilde,0); %"economy size" qr
switch model
    case 'linearX1'
        Beta=max(Y); %we do not estimate the slope and assume it=1
    case 'linearX2'
        Beta=max(Y); %we do not estimate the slope and assume it=1
    otherwise
        Beta=Gt\Q'*Ytilde;
end;
clear Gt;
clear Q;
phi=exp(xbest(1));
sigma2=exp(xbest(2));
tau2=exp(xbest(3));
logL=Lbest;


function [x,fval] =  runnestedSA(xc,n,F,Y,p,model,LB,UB,maxEvalsSA,nopars) 
% Nested "trick" to compute the objective function 
 %   options = saoptimset('PlotFcns',{@saplotbestx,...
 %               @saplotbestf,@saplotx,@saplotf},'MaxFunEvals',maxEvalsSA);
    options = saoptimset('Display','off','MaxFunEvals',maxEvalsSA);
    [x,fval] = simulannealbnd(@MLE2Nested,xc,LB,UB,options);
    function L=MLE2Nested(x)  %note the other parameters are not passed to MLE, they are local to the outer nested function
        %returns expression (2.5) in Lophaven et al (Dace manual, 2002)
        sigmaz2=exp(x(2))+exp(x(3));%this is a model free estimate, needed to make R a correlation matrix
        R=(exp(x(2))*exp(-(D*exp(x(1))).^p)+exp(x(3))*speye(n))/sigmaz2+(10+n)*eps*speye(n); %regularization (DACE toolbox manual eq. 5.4)
        C=chol(R,'lower');
        detR23=prod(diag(C).^(2/(n-nopars))); %computes det(R)^(2/n) in a much more numerically robust way
        Cinv=C\speye(n);
        clear C;
%         Ytilde=C\Y';
%         Ftilde=C\F;
        Ytilde=Cinv*Y;
        Ftilde=Cinv*F;
        clear Cinv;
        [Q,Gt]=qr(Ftilde,0); %"economy size" qr
        switch model
            case 'linearX1'
                Beta=max(Y); %we do not estimate the slope and assume it=1
            case 'linearX2'
                Beta=max(Y); %we do not estimate the slope and assume it=1
           otherwise
                Beta=Gt\Q'*Ytilde;%this could be solved faster by stepwise substitution starting from solving for Beta(k)
        end;
        clear Gt;
        clear Q;
        %REML estimates to correct bias due to mean parameters
        sigmaz2=(Ytilde-Ftilde*Beta)'*(Ytilde-Ftilde*Beta)/(n-nopars);       
        %CM=chol(F'*(R\F),'lower');
        CM=chol(Ftilde'*Ftilde,'lower');
        detCM=prod(diag(CM).^2);
        clear CM;
        logdetCM=log(detCM);
        L=log(detR23*sigmaz2)+logdetCM; % do return the log since usually the product is <1.0
    end
end

function R=Compute_R(x,n,p)
% isotropic powered exponential correlation function
sigmaz2=exp(x(2))+exp(x(3));%this is a model free estimate, needed to make R a correlation matrix
R=(exp(x(2))*exp(-(D*exp(x(1))).^p)+exp(x(3))*speye(n))/sigmaz2+(10+n)*eps*speye(n); %regularization (DACE toolbox manual eq. 5.4)
end

%MLE2 function if used by fminunc and when finding an initial solution

function L=MLE2(x,n,F,Y,p,model,nopars)
%returns expression (2.5) in Lophaven et al (Dace manual, 2002)
R=Compute_R(x,n,p);
C=chol(R,'lower');
Ytilde=C\Y;
Ftilde=C\F;
[Q,Gt]=qr(Ftilde,0); %"economy size" qr
switch model
    case 'linearX1'
        Beta=max(Y); %we do not estimate the slope and assume it=1
    case 'linearX2'
        Beta=max(Y); %we do not estimate the slope and assume it=1
   otherwise
        Beta=Gt\Q'*Ytilde;%this could be solved faster by stepwise substitution starting from solving for Beta(k)
end;
sigmaz2=(Ytilde-Ftilde*Beta)'*(Ytilde-Ftilde*Beta)/(n-nopars);
detR23=prod(diag(C).^(2/(n-nopars))); %computes det(R)^(2/(n-nopars)) in a much more numerically robust way
CM=chol(F'*(R\F),'lower');
detCM=prod(diag(CM).^2);
logdetCM=log(detCM);
L=log(detR23*sigmaz2)+logdetCM; % do return the log since usually the product is <1.0
end


end