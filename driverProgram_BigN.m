function [MSPx,MSPy,MSPz,MSPt,MSEonlyZ,MSPz_uv]=driverProgram_BigN(isomap)

% Driver program for simulation-estimation of non-euclidean GP models for
% 3D surface data. Accompanies the paper "Geodesic Gaussian Processes for
% Modeling Free-Form Surfaces" by E. del Castillo, B. Colosimo and Sam D. Tajbakhs
% Program returns vectors MSxxxx  with the mean
% square errors of the geodesic GP (x,y,z), the "euclidean Monge patch"
% z(x,y) (MSPonlyZ), and the "geodesic Monge patch" z(u,v) (MSPz_uv).
% Also returned are no_simulations*4 matrices phiV, sigma2V, and tau2V with the estimated
% covariance parameters associated with x(u,v), y(u,v), z(u,v) and z(x,y) models.
% Inputs are:    isomap (logical) if true, use isomap parameterization, if false, use ARAP
% parameterization

% To try a different dataset other than the Laser dataset, change line 58.

% Written by E. del Castillo and Sam D. Tajbakhsh, March 2013.

close all;

%% Setup optimizer (model fitting) settings
%  number of initial optimizer points, number of SA tries, model type and taper function for each coordinate model

nx = 1; nSAx = 200; model_x='interaction';
ny = 1; nSAy = 200; model_y='interaction';
nz = 1; nSAz = 200; model_z='quadratic';

% Set bounds for parameters phi,sigma2,tau2 (and p if also estimated)


LBno_px=[0.01,1e-6,1e-8];UBno_px=[20,.001,.001]; 
LBno_py=[0.01,1e-6,1e-8];UBno_py=[20,.001,.001]; 
LBno_pz=[0.01,1e-6,1e-8];UBno_pz=[20,.001,.001]; 

LBpx=[0.01,1e-8,1e-9,0.5];UBpx=[20,.001,.001,1.99];
LBpy=[0.01,1e-8,1e-9,0.5];UBpy=[20,.001,.001,1.99];
LBpz=[0.01,1e-8,1e-9,0.5];UBpz=[20,.001,.001,1.99];


%% Tolerances used in ARAP parameterization
epsARAP_nonoise=1e-5;
epsARAP_noise=1e-5; %we may wish a bit tighter here

%% Define p used in estimation (uncomment when estimate_p=false)
estimate_p=false; %true==>MLE_EBLUP_p is called; false==>MLE_EBLUP2 is called
if estimate_p==false
    p=1;
end;


%% Laser data analysis settings
%decimation=24; %used for laser data only; every 24 gives 398 points; every 10 gives 954 points
nfit=9000; %<--------------------------------------------------------------
npredict=635; %<-----------------------------------------------------------
 
%% Setup  counters for Laser data. Change file name if other data is desired.
h=open('fusionData.mat');   %<---------------------------------------------
points_all=zscore(h.Laser);%read and standardize all
[~, I, ~] = unique(points_all,'first','rows');
I = sort(I);
points_all = points_all(I,:);
points=points_all;
n=size(points,1);
indices=randsample(1:n,nfit+npredict);
for i=1:nfit
    points_fit(i,:)=points(indices(i),:);
end;
count=0;
for i=nfit+1:nfit+npredict
    count=count+1;
    points_predict(count,:)=points(indices(i),:);
end;
points=[points_fit;points_predict];
n=size(points,1); 


%% Find u-v parameterization
if isomap==false
% Use parameterization routine in G. Peyer's Graph toolbox library as an
% initial solution for ARAP
    
    dt=DelaunayTri(points(:,1:2));
    F=dt.Triangulation;  
%         m=makeMesh(points,F,[],points(:,1:2));
%         plotMesh(m);
    options.method='parameterization';
    options.laplacian='combinatorial';
    options.ndim=2;
    map = compute_parameterization(points,F,options);
    map=map';
else % use isomap2.m (renamed to avoid name conflicts) or use some other method
    map=isomap2(points,2); 
    %map=hlle(points,2);
end;
[~, map] = procrustes(points(:,1:2),map,'scaling',false,'reflection',false);
Xgrid=map;%no scaling done here as this mapping is just an input for ARAP
if isomap==false
% ARAP parameterization calculations
    [~,~,map]=ARAP_New(points,F,Xgrid,epsARAP_noise); 
    [~, map,~] = procrustes(points(:,1:2),map,'scaling',false,'reflection',false);
    Xgrid=map; % we must not scale the u,v coordinates or we'll lose the geodesic distances    
end;
    

close all;
h=figure(2);
scatter(Xgrid(:,1),Xgrid(:,2),'r');
xlabel('u');ylabel('v');
set(h,'color','white');

%% Model fitting
%Setup cros-validation in case function is unknown
Xgrid=map(1:nfit,:);
Xo=map(nfit+1:n,:);

[phi_x,p_x,tau2_x,sigma2_x,Beta_x,~,invCx,gammax,sStarx,indcTaperx] = MLE_EBLUP3(Xgrid,points_fit(:,1),model_x,nx,LBno_px,UBno_px,1,nSAx); %x=f(u,v)
[phi_y,p_y,tau2_y,sigma2_y,Beta_y,~,invCy,gammay,sStary,indcTapery] = MLE_EBLUP3(Xgrid,points_fit(:,2),model_y,ny,LBno_py,UBno_py,1,nSAy); %y=f(u,v)
[phi_z,p_z,tau2_z,sigma2_z,Beta_z,~,invCz,gammaz,sStarz,indcTaperz] = MLE_EBLUP3(Xgrid,points_fit(:,3),model_z,nz,LBno_pz,UBno_pz,1,nSAz); %z=f(u,v)
[phi_zfxy,p_zfxy,tau2_zfxy,sigma2_zfxy,Beta_zfxy,~,invCzfxy,gammazfxy,sStarzfxy,indcTaperzfxy] = MLE_EBLUP3(points_fit(:,1:2),points_fit(:,3),model_z,nz,LBno_pz,UBno_pz,1,nSAz); %z=f(x,y)

%% Predistion
%Predict the underlying x surface at the estimated u,v locations
[~,~,p_x,s2p_x]=MSEPredictions3(phi_x,p,tau2_x,sigma2_x,Beta_x,Xgrid,Xo,points_fit(:,1),model_x,invCx,gammax,sStarx,indcTaperx);
%Predict the underlying y surface at the estimated u,v locations
[~,~,p_y,s2p_y]=MSEPredictions3(phi_y,p,tau2_y,sigma2_y,Beta_y,Xgrid,Xo,points_fit(:,2),model_y,invCy,gammay,sStary,indcTapery);
%Predict the underlying z surface at the estimated u,v locations
[~,~,p_z,s2p_z]=MSEPredictions3(phi_z,p,tau2_z,sigma2_z,Beta_z,Xgrid,Xo,points_fit(:,3),model_z,invCz,gammaz,sStarz,indcTaperz);
%Predict the underlying z surface at the estimated x,y locations
[~,~,p_zfxy,s2p_zfxy]=MSEPredictions3(phi_zfxy,p,tau2_zfxy,sigma2_zfxy,Beta_zfxy,points_fit(:,1:2),points_predict(:,1:2),points_fit(:,3),model_z,invCzfxy,gammazfxy,sStarzfxy,indcTaperzfxy);


% %% Plot predicted points and compare vs. true NURBS and measurements
% h=figure(3);
% %change back to Xgrid-points so that we can print and evaluate predicted
% %points by cross validation
% points=points_predict;
% plot3(points(:,1),points(:,2),points(:,3),'*green');
% 
% hold on;
% title('');
% xlabel('x');ylabel('y');zlabel('z');   
% set(h,'color','white');
% % Print predictions
% plot3(p_x,p_y,p_z,'.k');
% hold on;
% 
% % Specify number of points for use in the plotting and statistic
% % gathering computations, depending on whether we are crossvalidating
% % (CMM) or using a 'true' simulated response
%  npoints=n;
%  plot3(points(:,1),points(:,2),p_zfxy,'sm');
%  xlabel('x');ylabel('y');zlabel('z');

%% Calculating MSPs
MSPx=norm(points_predict(:,1)-p_x,'fro')/sqrt(npredict);
MSPy=norm(points_predict(:,2)-p_y,'fro')/sqrt(npredict);
MSPz=norm(points_predict(:,3)-p_z,'fro')/sqrt(npredict);
MSPt=norm(points_predict-[p_x p_y p_z],'fro')/sqrt(npredict);
MSEonlyZ=norm([points_predict(:,1:2)  p_zfxy]-points_predict,'fro')/sqrt(npredict);
MSPz_uv=norm([points_predict(:,1:2) p_z]-points_predict,'fro')/sqrt(npredict);

end




end