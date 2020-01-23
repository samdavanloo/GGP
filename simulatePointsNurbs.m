function [X,points,TrueSurf,n2,Dgeo]=simulatePointsNurbs(no_points,srf,tau2,sigma2,p,phi,design,inputData,epsARAP,hole,isomap)
%function [X,points]=simulatePointsNurbs(no_points,srf,tau2,sigma2,pow,phi,design,inputData,epsARAP)
% Simulates NURBS surface 'srf', generating no_points^2 from a grid and adding a 
% 3D Gaussian Process with correlations computed over distances on the manifold.
% Uses toolbox "NURBS"
% Parameters:
    % no_points--sqrt(number of points to generate)
    % srf--NURBS structure representing the NURBS surface to sample
    % tau2,sigma2,p,phi--1x3 vectors of parameters for the powered exponential
    % spatial covariance function
    % hole--logical. If true, make a hole on surface (assumes no_points=30^2)
% Returns:
    % 3D noise-free points in TrueSurf   
    % points: TrueSurf + noise
    % X matrix with the x,y coordinates only (z not included)
    % n2: no. of simulated points, after points in a hole are deleted in
    % case hole==true 
    % Dgeo: vector with geodesic distances between all pairs of points on
    % surface (used only for democylind or sphere)
% Written by E. del Castillo, May 21, 2012
% History
% June 14 2012: changed sortrows(X,2) for sortrows(X). Also, read CMM data
% file

mean=[0 0 0];

% 1. create n^2 points (locations) x_i in domain. Note domain  is (0,1)^2

n=no_points; % setup n

if strcmp(design,'grid')
    %X=fullfact(repmat(n,1,2))/n;  %we used this in some of the simulations
    %in the paper
    X=(fullfact(repmat(n,1,2))-1)/(n-1); %we used this in the area computations
elseif strcmp(typeOfDOE,'latin')
    X=lhsdesign(n^2,2,'criterion','correlation');
    if dim==1 
        X=sort(X);
    else %dim=2
        X=sortrows(X);
    end;
end;

% 2. Next we generate points on the sinusoidal or NURBS surface
switch inputData
    case 'sphere'
        [x,y,z]=sphere(round(n*sqrt(2))-1);
        x=reshape(x,round(n*sqrt(2))^2,1);
        y=reshape(y,round(n*sqrt(2))^2,1);
        z=reshape(z,round(n*sqrt(2))^2,1);
        count=0;
        % keep upper half
        for i=1:round(n*sqrt(2))^2
            if z(i)>=0.9
                count=count+1;
                xupper(count)=x(i);
                yupper(count)=y(i);
                zupper(count)=z(i);
            end;
            % keep only a quarter
%             if (z(i)>=0)&(x(i)>=0)
%                 count=count+1;
%                 xupper(count)=x(i);
%                 yupper(count)=y(i);
%                 zupper(count)=z(i);
%             end;
        end;
        x=reshape(xupper,count,1);
        y=reshape(yupper,count,1);
        z=reshape(zupper,count,1);
        % Add noise 
        for i=1:count
            %note we add a tiny error since the parameterization routine has
            %problems when the surface is too 'regular'. Note that the actual
            %points we simulate further below we do not use these tiny errors
            %since we use TrueSurf and not TrueSurf2
            TrueSurfBefore(:,i) = [x(i); y(i) ;z(i)];              
            TrueSurf2Before(:,i) = TrueSurfBefore(:,i)+ mvnrnd(zeros(3,1),eye(3)*1e-8)'; 
        end; 
        %Delete multiple copies of point (0,0,1)
        count2=0;
        for i=1:count
            if not(x(i)==0&y(i)==0&z(i)==1)
                count2=count2+1;
                TrueSurf(:,count2)=TrueSurfBefore(:,i);
                TrueSurf2(:,count2)=TrueSurf2Before(:,i);
            end;
        end;  
        X=TrueSurf(1:2,:);
        n=round(sqrt(count2));
    case 'sinusoidal'
        %define desired number of cycles (waves)=w
        w=2.0;
        a=4; % a must be even because we want that sin(a*pi + pi/2)=1
        b=2*w+a; 
        d=(n-1)/(2*w); %denominator of the step size, step = pi/d
        [u,v]=ndgrid(a*pi+pi/2:pi/d:b*pi+pi/2,1:1:n);% n*n grid
        z=0.1*sin(u);
        %z=0.05*sin(u);
        mesh(u,v,z);
        u=reshape(u,n^2,1);
        v=reshape(v,n^2,1);
        z=reshape(z,n^2,1);
        u=(u-min(u))/range(u);
        v=(v-min(v))/range(v);
        % X only has the x-y coordinates
        X=[u v];
        for i=1:n^2
            %note we add a tiny error since the parameterization routine has
            %problems when the surface is too 'regular'. Note that the actual
            %points we simulate further below we do not use these tiny errors
            %since we use TrueSurf and not TrueSurf2
            TrueSurf(:,i) = [u(i); v(i) ;z(i)];
            TrueSurf2(:,i) = TrueSurf(:,i) + mvnrnd(zeros(3,1),eye(3)*1e-8)'; 
        end;          
    otherwise 
        for i=1:n^2  
         %  The order of u and v had to be changed to match the displayed
         %  surface
            TrueSurf(:,i) = nrbeval(srf,{X(i,2),X(i,1)});
            TrueSurf2(:,i) = TrueSurf(:,i) + mvnrnd(zeros(3,1),eye(3)*1e-8)';           
        end;
end;
% Compute geodesic distances between all points for a Cylinder and a sphere. 
switch inputData
    case 'democylind'
        % Assume radius=0.5 and height=1
        count=0;
        for i=1:size(TrueSurf,2)-1
            z1=TrueSurf(1,i);
            theta1=real(acos(TrueSurf(2,i)/0.5));
            for j=i+1:size(TrueSurf,2)
                count=count+1;
                %[i j]
                z2=TrueSurf(1,j);
                theta2=real(acos(TrueSurf(2,j)/0.5));
                Dgeo(count)=sqrt((z1-z2)^2+0.5^2*(theta1-theta2)^2);
            end;
        end;
    case 'sphere'
        % Assume radius=1
        count=0;
        for i=1:size(TrueSurf,2)-1
            p1=TrueSurf(:,i);
            for j=i+1:size(TrueSurf,2)
                p2=TrueSurf(:,j);
                count=count+1;
                %[i j]
                Dgeo(count)=acos(p1'*p2);
            end;
        end;
    otherwise
        Dgeo=[];
end;
% Add "hole" to surface (assumes n=30^2)
if hole
    TrueSurf(:,310:315)=[];
    TrueSurf2(:,310:315)=[];
    TrueSurf(:,334:339)=[];
    TrueSurf2(:,334:339)=[];
    TrueSurf(:,358:363)=[];
    TrueSurf2(:,358:363)=[];
    TrueSurf(:,382:387)=[];
    TrueSurf2(:,382:387)=[];
    TrueSurf(:,406:411)=[];
    TrueSurf2(:,406:411)=[];
end;
n2=size(TrueSurf,2);

% 3. Parameterize and unfold (flatten) the NURBS manifold using ARAP or
% Isomap
if isomap==false
    %F=delaunay(TrueSurf2(1:2,:)');
    F=delaunayn(TrueSurf2(1:2,:)',{''});
    %Compute areas of maximum gaussian curvature and plot them; we standardize
    %the data for this as procedure works better this way
    % [Umin,Umax,Cmin,Cmax,Cmean,Cgauss,Normal] = compute_curvature(zscore(TrueSurf'),F);
    % figure(5);%Cmax and Cmin are the principal curvatures at each point;Cmax(u,v)*Cmin(u,v)=Cgauss(u,v)
    % mesh(reshape(Cmax,n,n));hold on;mesh(reshape(Cmin,n,n));
    options.method='parameterization';
    options.laplacian='combinatorial';
    options.ndim=2;
    map = compute_parameterization(TrueSurf2,F,options);
    map=map';
else
    map=isomap2(TrueSurf2',2);%use isomap2.m to avoid name conflicts
end;
[~, map] = procrustes(TrueSurf2(1:2,:)',map,'scaling',false,'reflection',false);
if isomap==false
    [~,~,map]=ARAP_New(TrueSurf2',F,map,epsARAP); %note we do not ask too much in this parameterization since for very regular surfaces ARAP does not wrk well
    [~, map] = procrustes(TrueSurf2(1:2,:)',map,'scaling',false,'reflection',false);
end;
figure(2);
scatter(map(:,1),map(:,2),'r');

% 4. Simulate GRF over the parameterized 2D space
% 4.1 Create matrix of distances between points
Dgeodesics=squareform(pdist(map));
% 4.2 Compute Covariance matrix between all n^dim points
Sigma_x=sigma2(1)*exp(-(abs(Dgeodesics*phi(1)).^(p(1))))+eye(n2)*tau2(1);
Sigma_y=sigma2(2)*exp(-(abs(Dgeodesics*phi(2)).^(p(2))))+eye(n2)*tau2(2);
Sigma_z=sigma2(3)*exp(-(abs(Dgeodesics*phi(3)).^(p(3))))+eye(n2)*tau2(3);

% 5. Simulate realization of GRF
deltax=mvnrnd(ones(n2,1).*mean(1),Sigma_x);
deltay=mvnrnd(ones(n2,1).*mean(2),Sigma_y);
deltaz=mvnrnd(ones(n2,1).*mean(3),Sigma_z);
points=zeros(3,n2);
% Use real surface without the noise we added for parameterization purposes
% and GRF process
for i=1:n2
    points(:,i)=TrueSurf(:,i) +[deltax(i);deltay(i);deltaz(i)];
end;
points=points';%return a nx3 matrix
end