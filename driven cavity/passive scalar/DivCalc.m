function DIV = DivCalc(U,V)

global h Nx Ny
DIV  = nan(Nx-1,Ny-1);
i = 1:Nx-1;    j = 1:Ny-1;
DIV(i,j) = (U(i+1,j) - U(i,j))/h + (V(i,j+1) - V(i,j))/h;

end


