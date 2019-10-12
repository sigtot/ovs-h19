%close all;
%clear;

%% Create time
h       = 0.01;  % sample time (s)
N       = 50000; % number of samples
t       = 0:h:h*(N-1);

%% Insert true system
%m = @(n) meme(t(n), 20, 20*(2-exp(-0.01*(t(n)-20))), 20);
m = @(t) 15;
beta = 0.2;
k = 2;

lambda_1 = 30;
lambda_0 = 100;
Lambda = [1; lambda_1; lambda_0];

%% Define filters
[  ~,  ~,C_f_1,D_f_1]   = tf2ss([1],Lambda);
[  ~,  ~,C_f_2,D_f_2]   = tf2ss([1, 0],Lambda);
[A_f,B_f,C_f_3,D_f_3]   = tf2ss([1, 0, 0],Lambda);
% A_f and B_f is only needed once as it is the same all over. Hence tildes

%% Simulation MATLAB
% Define input as a function of t
%u       = @(n) 5*sin(2*t(n)) + 15;
u       = @(t) 50*sin(2*t) + 20*cos(3*t) +30*sin(2.5*t + 5) + 15;

% Memory allocation
x       = zeros(2, N);
x_z     = zeros(2, 1);
x_phi   = zeros(2, 1);
x_phi_aux = zeros(2, 1);
theta   = zeros(3, N);
U       = zeros(N-1,1);
masses  = zeros(1, N);
forget = 0.1;
x_2 = zeros(2,N);
x_1 = zeros(1,N);

% Initial estimates
%theta(:,1) = [k, beta/k, m(1)/k];
theta(:, 1) = [1, 1, 1];
masses(:,1) = m(1);

% Initial P matrix
gamma   = diag([100, 100, 100]);  

%% Ode45
y0 = [1; 1];
[~, x_2] = ode45(@(t,y) superfunc(t, y, m, u, beta), t, y0);
x_1 = 1/k * arrayfun(u, t)' + x_2(:, 1);

%% Estimation loop
% Main loop. Simulate using forward Euler (x[k+1] = x[k] + h*x_dot)
for n = 1:N-1
    % Generate z and phi by filtering known signals
    x_z_n           = x_z + (A_f*x_z + B_f*(x_1(n) + u(t(n)) - x_2(n, 1)))*h;   % u is unfiltered 'z'
    z               = C_f_1*x_z;                      % 1/Lambda * u

    x_phi_aux_n     = x_phi_aux + (A_f*x_phi_aux + B_f*(x_1(n) - x_2(n, 1))) * h; % *h???
    x_phi_n         = x_phi + (A_f*x_phi + B_f*x_2(n, 1))*h;
    phi             = [(C_f_1*x_phi_aux + D_f_1*(x_1(n) - x_2(n, 1)));
                       (C_f_2*x_phi + D_f_2*x_2(n, 1));
                       (C_f_3*x_phi + D_f_3*x_2(n, 1))];
    

    % Calculate estimation error
    epsilon         = (z - theta(:, n)'*phi)/(1 + 0.01*phi'*P(:, :, n)*phi);

    % Update law
    theta_dot       = gamma*epsilon*phi;
    theta(:, n+1)   = theta(:, n) + theta_dot*h;

    % Set values for next iteration
    x_phi           = x_phi_n;
    x_phi_aux       = x_phi_aux_n;
    x_z             = x_z_n;
    masses(:, n+1) = m(n+1);
end


% Plots
fig1 = figure(1);
subplot(3,1,1)
plot(t, theta(3,:) .* theta(1,:)); hold on
plot(t, masses(1,:)); hold off
ylabel('m')
title('Parameter estimates')
legend('estimate','true value')
grid
subplot(3,1,2)
plot(t, theta(2,:) .* theta(1, :)); hold on
plot([t(1), t(end)],[beta, beta]); hold off
ylabel('\beta')
legend('estimate','true value')
grid
subplot(3,1,3)
plot(t, theta(1,:)); hold on
plot([t(1), t(end)],[k, k]); hold off
ylabel('k')
grid
legend('estimate','true value')
xlabel('t [s]')
cd
%% Plot system
fig2 = figure(2);
plot(t, x_1); hold on;
plot(t, x_2(:, 1));
plot(t, x_2(:, 2));
plot(t, u(t)); hold off;
legend('y1', 'y2', 'y2_{dot}')
grid
