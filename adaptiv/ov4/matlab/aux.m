close all;
clear;

%%%%%%%%%%%% Problem 4.2 %%%%%%%%%%%%%%%%%
% Full state measurement and bounded u.
%
% Based on section 4.2.3 Vector Case, p. 156-162 in the book.
% (Also see section 5.2.2 for conditions on the input for the
% parameters to converge to their true values)
%
% This scripts shows how an adaptive law can be tested using MATLAB code only using ode45.
%
% This code uses the parallel model, but could just as
% easily had chosen the series-parallel.
% As noted on page 155. Parallel model is less sensitive to
% measurement noise, while the series-parallel model
% gives more design flexibility.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select input (0 or 1)
% 0: one frequency  u = 10*sin(2*t)
% 1: two frequencies u = 10*sin(2*t) + 7*cos(3.6*t)
w = 0;

% Second-order stable system
m = 20;
beta = 0.1;
k = 5;
u = 1;

lambda_1 = 1;
lambda_0 = 1;


a11 = 0;
a12 = 1;
a21 = -k/beta;
a22 = -beta/m;    %%%%
b1 = 0;
b2 = 1;

A = [a11 a12;
     a21 a22];

B = [b1;
     b2];

n = 2;



filter = s^2 + lambda_1*s + lambda_0;


z = u/filter;
phi = [


% Inital conditions
x0 = [0;0];

% Parameter estimator
A_m = diag([-10 -10]);
gamma_1 = 0.01; % Best so far: gamma_1 = 0.01, gamma_2 = 0.01
gamma_2 = 0.01;

% Initial estimates
x_hat_0 = [0;0];
m_hat_0 = 0;
beta_hat_0  = 0;
k_hat_0 = 0;
%A_hat_0 = 0*[-0.5,4;-3,0];
%B_hat_0 = 0*[0.5;3];
theta_star_0 = [m_hat_0; beta_hat_0, k_hat_0];

%% Simulate
tspan = [0 500];
%y0 = [x0;x_hat_0;A_hat_0(:);B_hat_0];
y0 = [x0;x_hat_0;theta_star_0];
[t,y] = ode45(@(t,y) parallel_model(t,y,A,B,gamma_1,gamma_2,w), tspan, y0);

%% Plot results

% eps
figure(3)
subplot(2,1,1)
plot(t,y(:,1)-y(:,3))
title('eps_1')
grid
subplot(2,1,2)
plot(t,y(:,2)-y(:,4))
title('eps_2')
xlabel('t [s]')
grid

% A_hat
figure(4)
subplot(2,2,1)
plot(t,y(:,5)); hold on
plot(t,a11*ones(length(t),1)); hold off
title('A_{11}')
grid
subplot(2,2,2)
plot(t,y(:,7)); hold on
plot(t,a12*ones(length(t),1)); hold off
title('A_{12}')
grid
subplot(2,2,3)
plot(t,y(:,6)); hold on
plot(t,a21*ones(length(t),1)); hold off
title('A_{21}')
grid
subplot(2,2,4)
plot(t,y(:,8)); hold on
plot(t,zeros(length(t),1)); hold off
title('A_{22}')
grid

% B_hat
figure(5)
subplot(2,1,1)
plot(t,y(:,9)); hold on
plot(t,1*ones(length(t),1)); hold off
title('B_1')
grid
subplot(2,1,2)
plot(t,y(:,10)); hold on
plot(t,b2*ones(length(t),1)); hold off
title('B_2')
grid
xlabel('t [s]')

