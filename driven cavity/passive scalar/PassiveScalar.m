clear; close all; clc;

% ==========================================================
%  2D incompressible lid-driven cavity flow (Harlow–Welch)
%  with passive scalar transport
%
%  - Navier–Stokes equations solved on a staggered grid
%    (Harlow–Welch formulation)
%  - Explicit 4th-order Runge–Kutta time integration
%  - Pressure projection at each RK stage to enforce
%    incompressibility
%  - Passive scalar advected by the velocity field and
%    diffused with constant diffusivity
%  - One-way coupling: the scalar does not affect the flow
% ==========================================================

global Nx Ny h hq Re kappa
global UNord USud UWest UEst VNord VSud VWest VEst

warning off

%% ------------------ Domain ------------------
Lx = 1; Ly = 1;
Nx = 80; Ny = 80;

x = linspace(0,Lx,Nx);
y = linspace(0,Ly,Ny);

h  = x(2)-x(1);      % grid spacing
hq = h^2;

%% ------------------ Parameters ------------------
Re    = 2000;        % Reynolds number
nu    = 1/Re;        % kinematic viscosity
kappa = 1e-3;        % scalar diffusivity
T     = 30;          % final time

%% ------------------ Staggered grids ------------------
U   = zeros(Nx,Ny-1);     % horizontal velocity (vertical faces)
V   = zeros(Nx-1,Ny);     % vertical velocity (horizontal faces)
P   = zeros(Nx-1,Ny-1);   % pressure (cell centers)
Phi = zeros(Nx-1,Ny-1);   % passive scalar (cell centers)

%% ------------------ Boundary conditions ------------------
% Velocity boundary conditions (lid-driven cavity)

UNord = 1; USud = 0;      % horizontal velocity
UWest = 0; UEst = 0;

VNord = 0; VSud = 0;      % vertical velocity
VWest = 0; VEst = 0;

% Apply boundary conditions to velocity fields
U(:,1)   = USud;
U(:,end) = UNord;
U(1,:)   = UWest;
U(end,:) = UEst;

V(:,1)   = VSud;
V(:,end) = VNord;
V(1,:)   = VWest;
V(end,:) = VEst;

%% ------------------ Initial scalar field ------------------
% Gaussian distribution centered in the cavity
[Xc,Yc] = meshgrid(x(2:end-1),y(2:end-1));
Phi = exp(-200*((Xc-0.5).^2 + (Yc-0.5).^2));

%% ------------------ Time step ------------------
Uref = 1;            % reference velocity

% CFL and diffusive stability constraints
C     = 1.2;
beta = 0.8;
Dt    = min([C*h/Uref, beta*hq*Re]);

Nt = round(T/Dt);    % number of time steps

%% ------------------ Pressure Laplacian ------------------
% Discrete Laplacian operator for pressure Poisson equation
G   = numgrid('S',Nx+1);
Lap = -delsq(G)/hq;

% Neumann boundary conditions for pressure
for i = 2:Nx
    j = 2;    k = G(i,j);   Lap(k,k) = Lap(k,k) + 1/hq;
    j = Ny;   k = G(i,j);   Lap(k,k) = Lap(k,k) + 1/hq;
end
for j = 2:Ny
    i = 2;    k = G(i,j);   Lap(k,k) = Lap(k,k) + 1/hq;
    i = Nx;   k = G(i,j);   Lap(k,k) = Lap(k,k) + 1/hq;
end

%% ------------------ Scalar monitoring ------------------
% Indices of the cavity center
ic = round((Nx-1)/2);
jc = round((Ny-1)/2);

% Scalar dissipation history
eps_phi = zeros(Nt,1);
time    = (0:Nt-1)*Dt;
% 
% video = VideoWriter('driven_cavity_scalar.mp4','MPEG-4');
% video.FrameRate = 10;
% open(video);
% 
% % --- Figura ad alta risoluzione
% figure(1)
% set(gcf,'Units','pixels',...
%         'Position',[100 100 1400 1200],...
%         'Color','w',...
%         'Renderer','opengl');
% % --- Frame di riferimento (dimensione fissa)
% exportgraphics(gcf,'temp_frame.png','Resolution',300)
% img_ref = imread('temp_frame.png');
% [NyF, NxF, ~] = size(img_ref);


%% ==========================================================
%                       TIME LOOP
% ==========================================================
for it = 1:Nt

   %% ===== RK4 with projection at each stage =====
        
        % ---------- STAGE 1 ----------
        [Fu1,Fv1] = RHS_HW(U,V);
        Fp1       = RHS_Phi(Phi,U,V);
        
        % ---------- STAGE 2 ----------
        U2 = U + 0.5*Dt*Fu1;
        V2 = V + 0.5*Dt*Fv1;
        
        % Pressure projection
        DIV = DivCalc(U2,V2)/Dt;
        P   = reshape(Lap\DIV(:),Nx-1,Ny-1);
        [Px,Py] = GradCalc(P);
        
        U2 = U2 - Dt*Px;
        V2 = V2 - Dt*Py;
        
        [Fu2,Fv2] = RHS_HW(U2,V2);
        Fp2       = RHS_Phi(Phi+0.5*Dt*Fp1,U2,V2);
        
        % ---------- STAGE 3 ----------
        U3 = U + 0.5*Dt*Fu2;
        V3 = V + 0.5*Dt*Fv2;
        
        DIV = DivCalc(U3,V3)/Dt;
        P   = reshape(Lap\DIV(:),Nx-1,Ny-1);
        [Px,Py] = GradCalc(P);
        
        U3 = U3 - Dt*Px;
        V3 = V3 - Dt*Py;
        
        [Fu3,Fv3] = RHS_HW(U3,V3);
        Fp3       = RHS_Phi(Phi+0.5*Dt*Fp2,U3,V3);
        
        % ---------- STAGE 4 ----------
        U4 = U + Dt*Fu3;
        V4 = V + Dt*Fv3;
        
        DIV = DivCalc(U4,V4)/Dt;
        P   = reshape(Lap\DIV(:),Nx-1,Ny-1);
        [Px,Py] = GradCalc(P);
        
        U4 = U4 - Dt*Px;
        V4 = V4 - Dt*Py;
        
        [Fu4,Fv4] = RHS_HW(U4,V4);
        Fp4       = RHS_Phi(Phi+Dt*Fp3,U4,V4);
        
        % ---------- FINAL UPDATE ----------
        U   = U + Dt*((Fu1+Fu4)/6 + (Fu2+Fu3)/3);
        V   = V + Dt*((Fv1+Fv4)/6 + (Fv2+Fv3)/3);
        Phi = Phi + Dt*((Fp1+Fp4)/6 + (Fp2+Fp3)/3);
        
        % ---------- FINAL PROJECTION ----------
        DIV = DivCalc(U,V)/Dt;
        P   = reshape(Lap\DIV(:),Nx-1,Ny-1);
        [Px,Py] = GradCalc(P);
        
        U = U - Dt*Px;
        V = V - Dt*Py;

    %% ---------- Scalar dissipation ----------
    % epsilon_phi = kappa * < |grad(phi)|^2 >
    [Phix, Phiy] = gradient(Phi,h);
    eps_phi(it) = kappa * mean(Phix(:).^2 + Phiy(:).^2);

    %% ---------- Output ----------
    if mod(it,20)==0
        divmax = max(abs(DivCalc(U,V)),[],'all');
        disp(['t=',num2str(it*Dt),'   max div=',num2str(divmax)])

        % Velocity at cell centers
        Uc = zeros(Nx,Ny);
        Vc = zeros(Nx,Ny);
        Uc(:,2:end-1) = 0.5*(U(:,1:end-1)+U(:,2:end));
        Vc(2:end-1,:) = 0.5*(V(1:end-1,:)+V(2:end,:));

           figure(1);  clf;
         % tiledlayout(2,2,'Padding','compact','TileSpacing','compact');


        % --------- (1) Velocity field ---------
        subplot(2,2,1)
        streamslice(x,y,Uc',Vc')
        % axis square
        axis equal
        axis([0 1 0 1])
        title('Velocity field')

        % --------- (2) Streamfunction ---------
        subplot(2,2,2)
        PSI = GivePsi(U,V);
        vneg = linspace(min(PSI(:)),0,10);
        vpos = linspace(0,max(PSI(:)),10);
        contour(x,y,PSI',vneg,'k'); hold on;
        contour(x,y,PSI',vpos,'r');
        axis square
        title('Streamfunction')

        % --------- (3) Passive scalar ---------
        subplot(2,2,3)
        contourf(x(2:end-1),y(2:end-1),Phi',20,'LineColor','none')
        colorbar
        axis square
        title('Passive scalar')

        % --------- (4) Scalar dissipation rate ---------
        subplot(2,2,4)
        plot(time(1:it),eps_phi(1:it),'LineWidth',1.5)
        grid on
        xlabel('Time')
        ylabel('\epsilon_\phi')
        title('Scalar dissipation rate')

        drawnow
        % exportgraphics(gcf,'temp_frame.png','Resolution',300)
        % img = imread('temp_frame.png');
        % 
        % % Forza dimensione costante
        % img = imresize(img,[NyF NxF]);
        % 
        % writeVideo(video, img);

    end
end
    % 
    % close(video);
    % disp('Video salvato correttamente!');

%% ==========================================================
%  Passive scalar RHS
% ==========================================================
function Fphi = RHS_Phi(Phi,U,V)
global h hq kappa

[NxP,NyP] = size(Phi);
Fphi = zeros(NxP,NyP);

% Loop ONLY over internal nodes
for i = 2:NxP-1
    for j = 2:NyP-1

        % Velocity at cell centers
        % (interpolated from staggered grid)
        u_c = 0.5*(U(i,j) + U(i+1,j));
        v_c = 0.5*(V(i,j) + V(i,j+1));

        % Central gradients of Phi
        dphidx = (Phi(i+1,j) - Phi(i-1,j))/(2*h);
        dphidy = (Phi(i,j+1) - Phi(i,j-1))/(2*h);

        % Convective term
        conv = u_c*dphidx + v_c*dphidy;

        % Laplacian (diffusion)
        lap = ( Phi(i+1,j) + Phi(i-1,j) + ...
                Phi(i,j+1) + Phi(i,j-1) - 4*Phi(i,j) )/hq;

        % Final RHS
        Fphi(i,j) = -conv + kappa*lap;
    end
end
end


