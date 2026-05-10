function [Fu,Fv] = RHS_HW(U,V)

global Nx Ny Re h hq UNord USud UWest UEst VNord VSud VWest VEst

Fu = zeros(size(U)); Fv = zeros(size(V));
Du = zeros(size(U)); Dv = zeros(size(V));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Termine Convettivo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPONENTE U

% Termine dx(Ix(u)*Ix(u))
i = 1:Nx-1;  j = 1:Ny-1;
Iu(i,j) = (U(i+1,j) + U(i,j))/2;              % Faccio la media ai centri cella
Iuq     = Iu.*Iu;                             % Faccio il quadrato sul posto
i = 2:Nx-1; j = 1:Ny-1;
Fu(i,j) = Fu(i,j) + (Iuq(i,j)-Iuq(i-1,j))/h;  % Calcolo la derivata sui nodi di U

% Termine dy(Iy(u)*Ix(v))
Iu = zeros(Nx,Ny); Iv = zeros(Nx,Ny);
Iu(:,Ny) = UNord;  Iu(:,1)  = USud; Iu(1,:)  = UWest;  Iu(Nx,:) = UEst;
Iv(:,Ny) = VNord;  Iv(:,1)  = VSud; Iv(1,:)  = VWest;  Iv(Nx,:) = VEst;
i = 2:Nx-1; j = 2:Ny-1;
Iu(i,j) = (U(i,j) + U(i,j-1))/2;              % Calcolo le medie ai vertici
Iv(i,j) = (V(i,j) + V(i-1,j))/2;
IuIv = Iu.*Iv;                                % Calcolo il prodotto sul posto
i = 2:Nx-1; j = 1:Ny-1;
Fu(i,j) = Fu(i,j) + (IuIv(i,j+1)-IuIv(i,j))/h;% Calcolo la derivata sui nodi di U
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPONENTE V

% Termine dx(Iy(u)*Ix(v))
i = 1:Nx-1; j = 2:Ny-1;
Fv(i,j) = Fv(i,j) + (IuIv(i+1,j)-IuIv(i,j))/h;% Calcolo la derivata sui nodi di V

% Termine dy(Iy(v)*Iy(v))
i = 1:Nx-1; j = 1:Ny-1;
Iv(i,j) = (V(i,j+1) + V(i,j))/2;              % Faccio la media ai centri cella
Ivq = Iv.*Iv;                                 % Faccio il quadrato sul posto
i = 1:Nx-1; j = 2:Ny-1;
Fv(i,j) = Fv(i,j) + (Ivq(i,j)-Ivq(i,j-1))/h;  % Calcolo la derivata sui nodi di U

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Termine Diffusivo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPONENTE U

i = 2:Nx-1; j = 1;
Du(i,j) = (U(i,j+1) + U(i+1,j) + U(i-1,j) + 2*USud  - 5*U(i,j))/hq;
i = 2:Nx-1; j = Ny-1;
Du(i,j) = (U(i,j-1) + U(i+1,j) + U(i-1,j) + 2*UNord - 5*U(i,j))/hq;
i = 2:Nx-1; j = 2:Ny-2;
Du(i,j) = (U(i,j-1) + U(i,j+1) + U(i+1,j) + U(i-1,j) - 4*U(i,j))/hq;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPONENTE V
i = 1;    j = 2:Ny-1;
Dv(i,j) = (V(i,j+1) + V(i+1,j) + V(i,j-1) + 2*VWest  - 5*V(i,j))/hq;
i = Nx-1; j = 2:Ny-1;
Dv(i,j) = (V(i,j+1) + V(i-1,j) + V(i,j-1) + 2*VEst   - 5*V(i,j))/hq;
i = 2:Nx-2; j = 2:Ny-1;
Dv(i,j) = (V(i,j-1) + V(i,j+1) + V(i+1,j) + V(i-1,j) - 4*V(i,j))/hq;


Fu = -Fu + Du/Re;
Fv = -Fv + Dv/Re;

end


