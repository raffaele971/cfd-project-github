clear; close all; clc;

% Homework 6 – Iterative methods for Poisson equation
% exercise at this link: https://www.kth.se/social/files/58ad6b81f276540de9208d1a/HW6.pdf
%
% Poisson problem:
%   ∇^2 p = f  in [0,1]x[0,1]
%   ∂p/∂n = 0  on boundary
% with f(x,y) = cos(pi*x)cos(pi*y)
%  
% Methods:
%   - Direct solve
%   - Gauss-Seidel
%   - SOR
%   - GMRES
%
% One pressure value is fixed: p(1,1) = 0


% PARAMETERS
Nx = 20;
Ny = 20;
Lx = 1; Ly = 1;

hx = Lx/Nx;
hy = Ly/Ny;
beta = hx/hy;

N = Nx*Ny;

tol = 1e-6;
maxIter = 10000;
omegaSOR = 1.9;

% GRID
x = ((1:Nx)-0.5)*hx;
y = ((1:Ny)-0.5)*hy;
[X,Y] = meshgrid(x,y);

% RHS f
f = cos(pi*X).*cos(pi*Y);
fvec = reshape(f',N,1);

% ASSEMBLE MATRIX L (with Neumann BC)
L = sparse(N,N);

idx = @(i,j) i + (j-1)*Nx;  %function created "on the fly"

for j = 1:Ny
    for i = 1:Nx
        k = idx(i,j);

        % Fix pressure at (1,1)
        if i==1 && j==1
            L(k,k) = 1;
            continue
        end

        L(k,k) = -2*(1/hx^2 + 1/hy^2);

        % i+1
        if i < Nx
            L(k,idx(i+1,j)) = 1/hx^2;
        else
            L(k,k) = L(k,k) + 1/hx^2;
        end

        % i-1
        if i > 1
            L(k,idx(i-1,j)) = 1/hx^2;
        else
            L(k,k) = L(k,k) + 1/hx^2;
        end

        % j+1
        if j < Ny
            L(k,idx(i,j+1)) = 1/hy^2;
        else
            L(k,k) = L(k,k) + 1/hy^2;
        end

        % j-1
        if j > 1
            L(k,idx(i,j-1)) = 1/hy^2;
        else
            L(k,k) = L(k,k) + 1/hy^2;
        end
    end
end


% DIRECT SOLUTION

p_direct = L\fvec;
Pdir = reshape(p_direct,Nx,Ny)';

figure;
surf(X,Y,Pdir,'EdgeColor','none');
title('Direct solution');
xlabel('x'); ylabel('y'); zlabel('p');
colorbar; view(45,30);


% GAUSS–SEIDEL

p = zeros(Nx,Ny);
resGS = [];

for m = 1:maxIter
    for j = 1:Ny
        for i = 1:Nx

            if i==1 && j==1
                p(1,1) = 0;
                continue
            end

            ip = min(i+1,Nx); im = max(i-1,1);
            jp = min(j+1,Ny); jm = max(j-1,1);

            p(i,j) = (1/(2*(1+beta^2))) * ...
                ( p(ip,j) + p(im,j) ...
                + beta^2*(p(i,jp)+p(i,jm)) ...
                - hx^2*f(i,j) );
        end
    end

    pvec = reshape(p',N,1);
    r = norm(L*pvec - fvec)/norm(fvec);
    resGS(end+1) = r;

    if r < tol
        break
    end
end

fprintf('GS iterations: %d\n',length(resGS));


% SOR

p = zeros(Nx,Ny);
resSOR = [];

for m = 1:maxIter
    for j = 1:Ny
        for i = 1:Nx

            if i==1 && j==1
                p(1,1) = 0;
                continue
            end

            ip = min(i+1,Nx); im = max(i-1,1);
            jp = min(j+1,Ny); jm = max(j-1,1);

            p_new = (1/(2*(1+beta^2))) * ...
                ( p(ip,j) + p(im,j) ...
                + beta^2*(p(i,jp)+p(i,jm)) ...
                - hx^2*f(i,j) );

            p(i,j) = (1-omegaSOR)*p(i,j) + omegaSOR*p_new;
        end
    end

    pvec = reshape(p',N,1);
    r = norm(L*pvec - fvec)/norm(fvec);
    resSOR(end+1) = r;

    if r < tol
        break
    end
end

fprintf('SOR iterations (ω=%.2f): %d\n',omegaSOR,length(resSOR));


% GMRES

[p_gmres,flag,~,~,reshist] = gmres(L,fvec,[],tol);


% CONVERGENCE PLOTS

figure;
semilogy(resGS,'LineWidth',1.5); hold on;
semilogy(resSOR,'LineWidth',1.5);
semilogy(reshist,'LineWidth',1.5);
grid on;
xlabel('Iteration');
ylabel('Relative residual');
legend('GS','SOR','GMRES','Location','northeast');
title('Convergence comparison');

