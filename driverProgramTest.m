function [MSPx,MSPy,MSPz,MSPt,MSEx,MSEy,MSEz,MSEt,MSEonlyZ,MSPz_uv,phiV,sigma2V,tau2V,pV]=driverProgramTest(inputData,filename,isomap)
% Driver program for the ITERATIVE simulation-estimation of non-euclidean GP models for
% 3D surface data. Assumes samml N (N<=1600 or so. For larger N's use driverProgram_BigN.mat).
% Accompanies the paper "Geodesic Gaussian Processes for
% Modeling Free-Form Surfaces" by E. del Castillo, B. Colosimo, and S. Tajbakhsh (2013).
% OUTPUTS: Program returns vectors MSP* of length no_simulations with the mean
% square prediction errors of the geodesic GP over each coordinate and for each of the 3 coordinates (MSPt), 
% the simulated MSE in each coordinate for comparison purposes (simualted measurement noise), 
% and the MSE for the standard or euclidean GP model for the
% ``heights", z(x,y) (MSPonlyZ), and the geodesic GP model for the heights, z(u,v) (MSPz_uv).
% Also returned are no_simulations*4 matrices phiV, sigma2V, and tau2V with the estimated
% covariance parameters associated with x(u,v), y(u,v), z(u,v) and z(x,y) models.
% MSE values and model parameter estimates saved in 'filename'
% Inputs are:  1) either a NURBS object name as per NURBS toolbox or the word "sinusoidal"
%                        (for the sinusoidal surface example) 
%                   2) an output filename (string)
%                   3) isomap (logical). If true, use isomap parameterization, if false, use ARAP
%                        parameterization
% Note: If parameter n_iter=1 (line 38), then the non-iterative GGP model fitting is performed.
% Written by E. del Castillo, Penn State U., IME Dept., November 2012

%% Setup optimizer (model fitting) settings
%  number of initial optimizer points, number of Simulated Annealing (SA) tries, and model type for each coordinate mean model

nx = 1; nSAx = 300; model_x='interaction'; 
ny = 1; nSAy = 300; model_y='interaction';
nz =1; nSAz = 300; model_z='intercept';  %note: used by GGP, z(u,v) and z(x,y)

% Set bounds for parameters phi,sigma2,tau2 (and p if also estimated)


    LBno_px=[0.01,1e-6,1e-8];UBno_px=[20,.001,.001]; LBpx=[0.01,1e-6,1e-8,0.5];UBpx=[20,.001,.001,1.99];
    LBno_py=[0.01,1e-6,1e-8];UBno_py=[20,.001,.001]; LBpy=[0.01,1e-6,1e-8,0.5];UBpy=[20,.001,.001,1.99];
    LBno_pz=[0.01,1e-6,1e-8];UBno_pz=[20,.001,.001]; LBpz=[0.01,1e-6,1e-8,0.5];UBpz=[20,.001,.001,1.99];


%% Define surface simulation settings
n_iter=1; % no. of iterations for the overall parameterization-fitting procedure
no_simulations=1;
n=20; %sqrt(number) of (u,v) points on the surface (number of points =n^2)
true_phi=[1 1 1];
true_sigma2=[0.0001,0.0001,0.0001]; %if these are all zero there is  no spatial correlation at all
true_tau2=[0.0001,0.0001,0.0001]; % nuggets
true_p=[1.0,1.0,1.0]; %p=1 exponential

%% Tolerances used in ARAP parameterization
epsARAP_nonoise=1e-5;
epsARAP_noise=1e-5; %we may wish a bit tighter here

%% Define p used in estimation (uncomment when estimate_p=false)
estimate_p=false; %true==>MLE_EBLUP_p is called; false==>MLE_EBLUP2 is called
if estimate_p==false
    p=1;
end;

%% Generate NURBS (if a NURBS; for sinusoidal or sphere function simply set variable srf equal to the string "sinusoidal" or "sphere")
switch inputData
    case 'demo4surf'
        srf=demo4surf;
    case 'democoons'
        srf=democoons;       
    case 'nrbtestsrf'
        srf=nrbtestsrf;     
    case 'democylind'
        srf=democylind;
    case 'democylindPatch'
        srf=democylindPatch;
    case 'sinusoidal'
        srf='sinusoidal';
    case 'sphere'
        srf='sphere';
end;
colormap(white);

% Specify number of points for use in the plotting and statistic gathering computations
    npoints=n^2;
    
%------------------------------------------------------------------------------------------
% Initialize matrices of covariance parameter estimates
phiV=zeros(no_simulations,n_iter,3);
tau2V=zeros(no_simulations,n_iter,3);
sigma2V=zeros(no_simulations,n_iter,3);
pV=zeros(no_simulations,n_iter,3);

%% MAIN LOOP; do for each simulation
for simulation=1:no_simulations
    close all;
    display(sprintf('SIMULATION # %5d',simulation));
    % Simulate measurement points on Nurbs, use ARAP since ISOMAP gives
    % trouble when simulating small spherical cap (need to fit GGP model
    % using ARAP too). This is because  the graph is not completely connected
    % (see isomap2.m)
    [Xgrid,points,TrueSurf,n2,Dgeo]=simulatePointsNurbs(n,srf,true_tau2,true_sigma2,true_p,true_phi,'grid',inputData,epsARAP_nonoise,false,false);
    close all;
    % Save original simulated points as we will iterate, and we need these
    % again later when computing MS's
    points_original=points;
    %% Parameterize and fit parametric models; do this n_iter times
    for it_number=1:n_iter  
        display(sprintf('Iteration # %5d',it_number));
        [p_x,p_y,p_z,Xgrid,phi,tau2,sigma2,pp]=Parameterize_Fit(points,points_original);
        phiV(simulation,it_number,:)=phi;
        tau2V(simulation,it_number,:)=tau2;
        sigma2V(simulation,it_number,:)=sigma2;
        pV(simulation,it_number,:)=pp;
        points=[p_x',p_y',p_z'];
        % Try ICP to align fitted points to original points
%         [R,T,~,~] = icp(points_original',points',15,'Matching','kDtree','Minimize','point');
%         points = R * points' + repmat(T,1,n^2);
%         points=points';
        % Empirical Mean square prediction error (no ICP registering/matching)
        MSPx(simulation,it_number)=norm(points(:,1)'-TrueSurf(1,:),'fro')/sqrt(npoints);
        MSPy(simulation,it_number)=norm(points(:,2)'-TrueSurf(2,:),'fro')/sqrt(npoints);
        MSPz(simulation,it_number)=norm(points(:,3)'-TrueSurf(3,:),'fro')/sqrt(npoints);
        MSPt(simulation,it_number)=norm(points'-TrueSurf,'fro')/sqrt(npoints);
         % Empirical mean square prediction error of z(u,v) 
        MSPz_uv(simulation,it_number)=norm([points_original(:,1:2) points(:,3)]-TrueSurf','fro')/sqrt(npoints);
        % Write iteration by iteration results to file
        dlmwrite(filename,[simulation,it_number,MSPx(simulation,it_number),MSPy(simulation,it_number),MSPz(simulation,it_number),MSPt(simulation,it_number),MSPz_uv(simulation,it_number),reshape(phiV(simulation,it_number,:),1,3),reshape(sigma2V(simulation,it_number,:),1,3),reshape(tau2V(simulation,it_number,:),1,3),reshape(pV(simulation,it_number,:),1,3)],'-append');
    end;
     
    %% Fit model z=f(x,y) (a standard euclidean GP on the "heights") in order to compare to our approach
    display(sprintf('*****Now fitting z=f(x,y) in order to compare...'));
    if estimate_p==false
        [ phi,p,tau2,sigma2,Beta,logL] = MLE_EBLUP2(points_original(:,1:2),points_original(:,3),model_z,nz,LBno_pz,UBno_pz,p,nSAz);
    else
        [ phi,p,tau2,sigma2,Beta,~,i_repeats] = MLE_EBLUP_p(points_original(:,1:2),points_original(:,3),model_z,nz,LBpz,UBpz,nSAz);
    end;
    phiV_fxy(simulation)=phi;
    tau2V_fxy(simulation)=tau2;
    sigma2V_fxy(simulation)=sigma2;
    pV_fxy(simulation)=p;
    % Now predict zhat=f(x,y) at the observed values
    [~,~,p_zfxy]=MSEPredictions(phi,p,tau2,sigma2,Beta,points_original(:,1:2),points_original(:,1:2),points_original(:,3),model_z);

    %% Plot predicted points and compare vs. true NURBS and measurements
    
    h=figure(3);
    %Generate and plot NURBS again
    switch inputData
        case 'demo4surf'
            srf=demo4surf;
            colormap(white);
        case 'democoons'
            srf=democoons;
            colormap(white);
        case 'nrbtestsrf'
            srf=nrbtestsrf;
            colormap(white);
        case 'democylind'
            srf=democylind;
            colormap(white);
        case 'democylindPatch'
            srf=democylindPatch;
            colormap(white);
        case 'sinusoidal'
            u=reshape(TrueSurf(1,:),n,n);
            v=reshape(TrueSurf(2,:),n,n);
            z=reshape(TrueSurf(3,:),n,n);
            mesh(u,v,z);
            colormap(gray);
        case 'sphere'
            %we do nothing and just plot the points
    end;
    hold on;
    title('');
    xlabel('x');ylabel('y');zlabel('z');   
    set(h,'color','white');
    % Print predictions
    plot3(p_x,p_y,p_z,'.k');
    hold on;  
    plot3(points(:,1),points(:,2),p_zfxy,'sm');
    xlabel('x');ylabel('y');zlabel('z');
    plot3(TrueSurf(1,:),TrueSurf(2,:),TrueSurf(3,:),'*green');
    title('(Xhat(u,v),Yhat(u,v),Zhat(u,v)) --dark--and true surface--green--');
    % % Do ICP of predictions w.r.t true surface
    % [R,T,ER,t] = icp(TrueSurf,[p_x;p_y;p_z],15,'Matching','kDtree','Minimize','point');
    % preds = R * [p_x;p_y;p_z] + repmat(T,1,n^2);
    % plot3(preds(1,:)',preds(2,:)',preds(3,:)','.r');

    % Print observations
    %plot3(points(:,1),points(:,2),points(:,3),'*green');
    h=figure(4);
    hold on;
    title('Xhat(u,v)--black-- and X(u,v)--green');
    plot3(Xgrid(:,1),Xgrid(:,2),p_x,'.k');
    plot3(Xgrid(:,1),Xgrid(:,2),points(:,1),'*green');
    plot3(Xgrid(:,1),Xgrid(:,2),zeros(size(Xgrid,1),1),'ored');
    xlabel('u');ylabel('v');zlabel('x(u,v)');
    set(h,'color','white');
    h=figure(5);
    hold on;
    title('Yhat(u,v)--black-- and Y(u,v)--green');
    plot3(Xgrid(:,1),Xgrid(:,2),p_y,'.k');
    plot3(Xgrid(:,1),Xgrid(:,2),points(:,2),'*green');
    plot3(Xgrid(:,1),Xgrid(:,2),zeros(size(Xgrid,1),1),'ored');
    xlabel('u');ylabel('v');zlabel('y(u,v)');
    set(h,'color','white');
    h=figure(6);
    hold on;
    title('Zhat(u,v)--black-- and Z(u,v)--green');
    plot3(Xgrid(:,1),Xgrid(:,2),p_z,'.k');
    plot3(Xgrid(:,1),Xgrid(:,2),points(:,3),'*green');
    plot3(Xgrid(:,1),Xgrid(:,2),zeros(size(Xgrid,1),1),'ored');
    xlabel('u');ylabel('v');zlabel('z(u,v)');
    set(h,'color','white');
    

    % Mean Square prediction error if we only fit a model to z(x,y) but use x
    % and y as measured 
    MSEonlyZ(simulation)=norm([points_original(:,1:2)  p_zfxy']-TrueSurf','fro')/sqrt(npoints);

    % Note: the follwoing was commented out since aligning using ICP
    % was found to have no effect or even a worsening effect.

    % MSE if fitting Z(x,y) after ICP
    % [R,T,ER,t] = icp(TrueSurf,[points(:,1)';points(:,2)';p_zfxy],15,'Matching','kDtree','Minimize','point');
    % p_zfxy_preds = R * [points(:,1)';points(:,2)';p_zfxy] + repmat(T,1,n^2);
    % MSEonlyZ_icp=norm(p_zfxy_preds - TrueSurf);

    % Recompute Empirical MS prediction errors
    % MSPx_icp=norm(preds(1,:)-TrueSurf(1,:));
    % MSPy_icp=norm(preds(2,:)-TrueSurf(2,:));
    % MSPz_icp=norm(preds(3,:)-TrueSurf(3,:));
    % MSPt_icp=norm(preds-TrueSurf);
    % 
    % Now do ICP of points w.r.t true surface, to be fair
    % [R,T,ER,t] = icp(TrueSurf,points',15,'Matching','kDtree','Minimize','point');
    % points = R * points' + repmat(T,1,n^2);
    % NOTE: points is now 3 * n^2
    %points=points';
    % Compute the mean square error of the observed points (mean square error per point)
    MSEx(simulation)=norm(points_original(:,1)'-TrueSurf(1,:),'fro')/sqrt(npoints);
    MSEy(simulation)=norm(points_original(:,2)'-TrueSurf(2,:),'fro')/sqrt(npoints);
    MSEz(simulation)=norm(points_original(:,3)'-TrueSurf(3,:),'fro')/sqrt(npoints);
    MSEt(simulation)=norm(points_original'-TrueSurf,'fro')/sqrt(npoints);
    
    %% Write results in file

    dlmwrite(strcat(filename,'2'),[simulation,MSEx(simulation),MSEy(simulation),MSEz(simulation),MSEt(simulation),MSEonlyZ(simulation),phiV_fxy(simulation),sigma2V_fxy(simulation),tau2V_fxy(simulation),pV_fxy(simulation)],'-append');

end; % MAIN LOOP

%% Nested function that is called to compute the u,v's and predict x,y,z
    function [p_x,p_y,p_z,Xgrid,phiV_it,tau2V_it,sigma2V_it,pV_it]=Parameterize_Fit(points,points_original)
        %% Find u-v parameterization; here we use "points", i..e., the mos recent predicted (reconstructued) 3D coordinates
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

        % Attempt to fix the ARAP parameterization use either 'stretch' or
        % 'energy' (not used--commented out)
    %     [new_map,oldE,newE]= computeEnergies(points,F,map,round((n^2)*0.15),'energy');    
    %     close all;
    %     figure(1);
    %     hold on;
    %     dt_uv=DelaunayTri(map);
    %     OldSurfaceArea=compute_Surface_Area(dt_uv);
    %     scatter(Xgrid(:,1),Xgrid(:,2),'ob');
    %     triplot(dt_uv, dt_uv.X(:,1), dt_uv.X(:,2));
    %     hold on;   
    %     Xgrid=new_map;
    %     dt_uv_new=DelaunayTri(new_map);
    %     NewSurfaceArea=compute_Surface_Area(dt_uv_new);
    %     scatter(new_map(:,1),new_map(:,2),'.r');
    %     triplot(dt_uv_new, dt_uv_new.X(:,1), dt_uv_new.X(:,2));
    %     title(['Old Energy is: ',num2str(oldE),'  New energy: ',num2str(newE),'  Old Area: ',num2str(OldSurfaceArea),'  New Area: ',num2str(NewSurfaceArea)]);

        close all;
        % Plot euclidean (u,v) plane distance vs. geodesic surface distances if cylinder or sphere case to
        % check how good the near isometry is
%         if (strcmp(inputData,'democylind')||strcmp(inputData,'sphere'))
%                 distEuclid=pdist(Xgrid);
%                 D=squareform(distEuclid);
%                 count=0;
%                 for i=1:size(Xgrid,1)-1
%                     for j=i+1:size(Xgrid,1)
%                         count=count+1;
%                         Deuc(count)=D(i,j);
%                     end;
%                 end;              
%                 h=figure(1);
%                 scatter(Dgeo,Deuc,'.k');
%                 hold on;
%                 title(['Correlation coefficient is ',num2str(real(corr(Dgeo,Deuc)))]);
%                 xlabel('Euclidean distance on D');ylabel('Geodesic distance on S');
%                 set(h,'color','white');
%         end;
%         h=figure(2);
%         scatter(Xgrid(:,1),Xgrid(:,2),'r');
%         xlabel('u');ylabel('v');
%         set(h,'color','white');

        %% Model fitting and Prediction: Estimate 3 GP parametric surface models. 
        % Use original points (points_original) as these are the ones we
        % wish to model at the estimated (u,v) locations
        % First estimate x coordinate parametric surface
        if estimate_p==false
            [ phi,p,tau2,sigma2,Beta,logL] = MLE_EBLUP2(Xgrid,points_original(:,1),model_x,nx,LBno_px,UBno_px,p,nSAx);
        else
            [ phi,p,tau2,sigma2,Beta,~,i_repeats] = MLE_EBLUP_p(Xgrid,points(:,1),model_x,nx,LBpx,UBpx,nSAx);
        end;
        phiV_it(1)=phi;
        tau2V_it(1)=tau2;
        sigma2V_it(1)=sigma2;
        pV_it(1)=p;
        Beta_x=Beta; 
        % Now predict the underlying x surface at the estimated u,v locations
        [~,~,p_x,s2p_x]=MSEPredictions(phi,p,tau2,sigma2,Beta,Xgrid,Xgrid,points_original(:,1),model_x);
        % Estimate y coordinate parametric surface
        if estimate_p==false
            [ phi,p,tau2,sigma2,Beta,logL] = MLE_EBLUP2(Xgrid,points_original(:,2),model_y,ny,LBno_py,UBno_py,p,nSAy);
        else
            [ phi,p,tau2,sigma2,Beta,~,i_repeats] = MLE_EBLUP_p(Xgrid,points(:,2),model_y,ny,LBpy,UBpy,nSAy);
        end;
        phiV_it(2)=phi;
        tau2V_it(2)=tau2;
        sigma2V_it(2)=sigma2;
        pV_it(2)=p;
        Beta_y=Beta;
        % Now predict the underlying y surface at the estimated u,v locations
        [~,~,p_y,s2p_y]=MSEPredictions(phi,p,tau2,sigma2,Beta,Xgrid,Xgrid,points_original(:,2),model_y);

        % Estimate z coordinate parametric surface
        if estimate_p==false
            [ phi,p,tau2,sigma2,Beta,logL] = MLE_EBLUP2(Xgrid,points_original(:,3),model_z,nz,LBno_pz,UBno_pz,p,nSAz);
        else
            [ phi,p,tau2,sigma2,Beta,~,i_repeats] = MLE_EBLUP_p(Xgrid,points_original(:,3),model_z,nz,LBpz,UBpz,nSAz);
        end;
        phiV_it(3)=phi;
        tau2V_it(3)=tau2;
        sigma2V_it(3)=sigma2;
        pV_it(3)=p;
        Beta_z=Beta;
        % Now predict the underlying z surface at the estimated u,v locations
        [~,~,p_z,s2p_z]=MSEPredictions(phi,p,tau2,sigma2,Beta,Xgrid,Xgrid,points(:,3),model_z);
    end
end