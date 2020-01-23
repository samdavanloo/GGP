function [x,t,vt]=ARAP_New(x_0,t_0,vt_0,tolerance)
% ARAP perameterization 
% Select correct .obj file, which contains texture coordinates
% to be the initial guess of lcal/global algorithm. Input the 
% file name as the first parameter of ARAP function.
% Set the tolerance of convergence. For most common case, 
% tolerance = 0.001 is enough.

% Modified by EDC, PSU, May 2012: there is no need for an .obj object for
% input/output


x=x_0;
t=t_0;
vt=vt_0;
% fullname=[fname,'.obj'];
% [x,t,vt]=LoadOBJ(fullname);

%%%%%%%%%%%%%%%%%%%% Plot Initial Guess %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure
% trimesh(t,vt(:,1),vt(:,2));
% axis equal;
% title('Initial Guess');

%%%%%%%%%%%%%%%%%%%% Pre-Computations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EV=CalEdgeVectors(x,t);   %Calculate edge vectors of all the triangles
C=CalCots(x,t);   %Calculate cot weights of each angle in each triangle 
L=laplacianARAP(x,t,C);  %Compute cot Laplacian of the mesh
L(1,size(L,1)+1)=1;
L(size(L,1)+1,1)=1;
% Lc=chol(L);
% Linv=L'*Lc;
%Linv = inv(L);  % do not use inv
[L1,U]=lu(L);
%%%%%%%%%%%%%%%%%%%% Local/Global Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%
E=-1;
Epre=0;
iterations=0;

while (abs(Epre-E)>tolerance)&&(iterations<200)     %Iteration stops when the energy converges 
    iterations=iterations+1;    %to the tolerance, which is specified by
    Epre=E;                     %user.
    R=ARAP_Local(vt,t,EV,C);
    vt=ARAP_Global(EV,L1,U,t,C,R);
    E=CalRigidEnergy(EV,t,vt,C,R);
%%%%%%%%%% Plot iteration results (nice but slows down everything A LOT)%%%%%%%%%%%%%%%%%%%%%%%%
%     figure(20);
%     trimesh(t,vt(:,1),vt(:,2));
%     axis equal;
%     title1=num2str(iterations);
%     title2=[title1,' iterations','error=',num2str(abs(Epre-E))];
%     title(title2);
%     
end

%%%%%%%% Show para result %%%%%%%%%%%%%%%%%%%%%
figure
trimesh(t,vt(:,1),vt(:,2));
axis equal;
title1=num2str(iterations);
title2=[title1,' Iterations'];
title(title2);


%%%%%%%%%% Output the .obj file with texture coordinates to the
%%%%%%%%%% disk, called 'originalname_ARAP.obj'
% newfilename=[fname,'_ARAP.obj'];
% WriteOBJ(newfilename, x, t, vt);