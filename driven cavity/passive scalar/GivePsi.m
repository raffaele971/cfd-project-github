
function PSI = GivePsi(U,V)

% Computes the streamfunction PSI from a staggered velocity field
% (Harlow–Welch grid) by solving:
%     Laplacian(PSI) = -vorticity

global h

[Nx,~] = size(U);        % U : Nx x (Ny-1)
[~,Ny] = size(V);        % V : (Nx-1) x Ny

%% --- Velocity derivatives ---
Uy = zeros(Nx,Ny);
Vx = zeros(Nx,Ny);

i = 2:Nx-1;
j = 2:Ny-1;

Uy(i,j) = (U(i,j)   - U(i,j-1))/h;
Vx(i,j) = (V(i,j)   - V(i-1,j))/h;

%% --- Vorticity at cell centers ---
ZITA = Uy - Vx;
ZITA = ZITA(2:end-1,2:end-1);     % (Nx-2) x (Ny-2)

nx = Nx-2;
ny = Ny-2;

zita = ZITA(:);

%% --- Poisson problem for streamfunction ---
G      = numgrid('S',nx+2);
LapPsi = -delsq(G)/(h^2);

Psi = reshape(LapPsi \ zita, nx, ny);

%% --- Add homogeneous Dirichlet BCs (Psi = 0 on walls) ---
PSI = zeros(Nx,Ny);
PSI(2:end-1,2:end-1) = Psi;

end
