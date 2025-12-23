clear; close all; clc;

% ==============================================================
%  1D Convection-Diffusion Equation Solver with Runge-Kutta Integration
% ==============================================================
% This MATLAB code solves the 1D convection-diffusion equation:
% 
%   ∂ϕ/∂t + a ∂ϕ/∂x - k ∂²ϕ/∂x² = 0
%
% The solution is computed using a finite difference method with:
% - Central difference scheme (5 points) for diffusion term
% - Upwind scheme (4 points) for convection term
%  The spatial domain [0, L] is discretized using a Chebyshev grid,
% The Chebyshev nodes are defined as:
% 
%   x_j = L/2 * (1 - cos(θ_j)),  for j = 1, 2, ..., N
% 
% where θ_j are the Chebyshev nodes given by:
% 
%   θ_j = (j-1) * π / (N-1)  for j = 0, 1, 2, ..., N-1
%
% The time integration is carried out using a 3-stage explicit 
% Runge-Kutta method with the parameter 'alpha'. The code provides 
% the following features:
% 
% 1. Time Evolution: The solution is updated and visualized over 
%    time in real-time using `drawnow`.
% 2. Stability Map: A stability analysis of the numerical scheme 
%    is performed in the (C, β) parameter space, where C is the 
%    Courant number and β is the diffusion coefficient.
% 
% The code can be run as a whole, or individual sections can be 
% commented/uncommented depending on the task at hand.
%
% Author: Raffaele Riccardi
% ==============================================================

% parameters
L = 1;
N = 80;
a = 1;
k = 1e-3;
alpha = 0.5;

dt   = 5e-4;
Tend = 1;
Nt   = round(Tend/dt);

dx   = L/(N-1);
C    = a*dt/dx;
beta = k*dt/dx^2;

fprintf('C = %.3f   beta = %.3f\n',C,beta);

% Chebyshev grid
j = 0:N-1;
theta = j*pi/(N-1);
x = L/2*(1 - cos(theta))';

% initial condition
phi = exp(-150*(x - L/2).^4);

% differential operators
D1 = zeros(N);
D2 = zeros(N);

for i = 1:N
    im1 = mod(i-2,N)+1;
    im2 = mod(i-3,N)+1;
    im3 = mod(i-4,N)+1;
    ip1 = mod(i,N)+1;
    ip2 = mod(i+1,N)+1;

    % upwind 4 points (a>0)
    D1(i,i)   =  11/(6*dx);
    D1(i,im1) = -18/(6*dx);
    D1(i,im2) =  9/(6*dx);
    D1(i,im3) = -2/(6*dx);

    % central diffusion- 5 points
    D2(i,i)   = -30/(12*dx^2);
    D2(i,ip1) =  16/(12*dx^2);
    D2(i,im1) =  16/(12*dx^2);
    D2(i,ip2) =  -1/(12*dx^2);
    D2(i,im2) =  -1/(12*dx^2);
end

A = -a*D1 + k*D2;

% Runge-Kutta coefficients
b1 = 1/2 - 1/(6*alpha);
b2 = 1/(6*alpha*(1-alpha));
b3 = (2-3*alpha)/(6*(1-alpha));

%plot
figure;
h = plot(x,phi,'LineWidth',2);
axis([0 L -0.1 1.1]);
grid on;
xlabel('x');
ylabel('\phi');
title('t = 0');

drawnow;

% time loop
time = 0;

for n = 1:Nt
    k1 = A*phi;
    k2 = A*(phi + alpha*dt*k1);
    k3 = A*(phi + dt*((1+(1-alpha)/(alpha*(3*alpha-2)))*k1 ...
        - (1-alpha)/(alpha*(3*alpha-2))*k2));

    phi = phi + dt*(b1*k1 + b2*k2 + b3*k3);
    time = time + dt;

    % update plot
    if mod(n,5) == 0   % update every five step
        set(h,'YData',phi);
        title(sprintf('t = %.3f',time));
        drawnow;
        % pause(0.01)   
    end
end

