clear all; close all;

varying_mass = false;

%% Parameters
rho_0 = 2;
rho_1 = 0.1;
beta_forget = 0.1;
R_0 = 5;

k_0 = 1;
m_0 = 5;
beta_0 = 0.3;

%% Model
% System model given by m*y'' + beta*y' + k*y = u
m = 20;
beta = 0.1;
k = 5;

% Define state space model x' = Ax + Bu, where x = [y', y]^T
A = [0, 1; -k/m, -beta/m];
B = [0; 1/m];

%% Filter
% Second order stable filter
% lambda = (s+1)^2 = s^2 +2s +1
lambda = [1, 2, 1];

% Realize filters using three state space models. Remember, we need three filter s^2/(s^2 +2s +1), s/(s^2 +2s +1) og 1/(s^2 +2s +1)
% to generate phi. 
[~,~,C_f_1,D_f_1] = tf2ss([0, 0, 1],lambda);
[~,~,C_f_2,D_f_2] = tf2ss([0, 1, 0],lambda);
[A_f,B_f,C_f_3,D_f_3] = tf2ss([1, 0, 0],lambda);


%% Simulation 
h = 0.01;                    % sample time (s)
simTime = 600;               % simulation duration in seconds
N  = simTime/h;              % number of samples

% Memory allocation (more efficient to do this a priori)
t = zeros(1, N);
x = zeros(2, N);
U = zeros(1, N);
theta = zeros(2, N); theta(:, 1) = [beta_0; k_0];
rho = zeros(1, N); rho(1) = k_0;
theta_star = zeros(2, N); theta_star(:, 1) = [beta, m];
rho_star = zeros(1,N); rho_star(:, 1) = k;
x_z = zeros(size(A_f,1), 1);
x_phi = zeros(size(A_f,1), 1);
x_rho = 0;
x_u_filt = zeros(size(A_f, 1), 1);

gamma_k = 0.01;
Gamma = diag([1; 5]);

% NB: All initial conditions are zero

% Main simulation loop
for n = 1:N-1
    % Generate input
    t(n+1) = n*h;
    u = 10*sin(2*t(n)) + 10*cos(3.6*t(n)) + 20*cos(1*t(n)); 
    U(n) = u;
    
    % Changing mass after t = 20s
    if(varying_mass && t(n) > 20)
        m = 20*(2-exp(-0.01*(t(n) - 20)));
        A = [0, 1; -k/m, -beta/m];
        B = [0; 1/m];
    end
    
    % Save actual model parameters at timestep for comparison later
    theta_star(:, n+1) = [beta, m];
    rho_star(:, n+1) = k;
    
    % Simulate actual system
    x_dot = A*x(:, n) + B*u;
    x(:, n+1) = x(:, n) + h*x_dot;
    y = x(1, n);
    
    % Calculate z 
    x_z_n = x_z + (A_f*x_z + B_f*y)*h; % Euler integration
    z = C_f_1*x_z; % Fetch output
    
    % Calculate phi
    x_phi_n = x_phi + (A_f*x_phi + B_f*(-y))*h; % Euler integration
    phi_filt = [C_f_2*x_phi + D_f_2*(-y); C_f_3*x_phi + D_f_3*(-y);]; % Fetch output
    
    % Calculate z_1 = u_filt
    x_u_filt_n = x_u_filt + (A_f*x_u_filt + B_f*u)*h; % Euler integration
    u_filt = C_f_1*x_u_filt; % Fetch output
    
    % Calculate z_hat
    xi_filt = theta(:, n)'*phi_filt + u_filt;
    z_hat = rho(:, n)*xi_filt;
    
    % Estimation error
    epsilon = z - z_hat;
    
    % Propagate parameter estimates using Euler intagration
    theta_dot = Gamma*epsilon*phi_filt;
    theta(:, n+1) = theta(:, n) + theta_dot*h;
    
    rho_dot = gamma_k * epsilon * xi_filt;
    rho(:, n+1) = rho(:, n) + rho_dot * h;
    
    % Set variables for next iteration
    x_phi = x_phi_n;
    x_z = x_z_n;
    x_u_filt = x_u_filt_n;
end

figure(1);
hold on;
plot(t, theta);
plot(t, 1./rho)
plot(t, theta_star, '--')
plot(t, rho_star, '--')
grid on;
legend('\beta', 'm', 'k');
title("Parameter estimates for bilinear form");

figure(2);
hold on;
plot(t, x);
grid on;
title("MSD system simulation");