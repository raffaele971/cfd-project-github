function [Px,Py] = GradCalc(P)

global h Nx Ny
Px = zeros(Nx,Ny-1);  Py = zeros(Nx-1,Ny);

i = 2:Nx-1; j = 1:Ny-1;
Px(i,j) = (P(i,j) - P(i-1,j))/h;

i = 1:Nx-1; j = 2:Ny-1;
Py(i,j) = (P(i,j) - P(i,j-1))/h;

end

