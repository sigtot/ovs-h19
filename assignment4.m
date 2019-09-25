close all;
clear;

%% Create time
h       = 0.01;  % sample time (s)
N       = 50000; % number of samples
t       = 0:h:h*(N-1);

%% Insert true system
m = @(n) 20*(2-exp(-0.01*(t(n)-20)));
beta = 0.1;
k = 5;
A = @(n) [0       1
          -k/m(n) -beta/m(n)]; % <-- Fill in blanks
B = @(n) [0
          1/m(n)];               % <-- Fill in blanks

 lambda_1 = 1;
 lambda_0 = 1;
 Lambda = [1; lambda_1; lambda_0];

%% Define filters
[  ~,  ~,C_f_1,D_f_1]   = tf2ss([1],Lambda);           % <-- Fill in blanks
[  ~,  ~,C_f_2,D_f_2]   = tf2ss([1, 0],Lambda);           % <-- Fill in blanks
[A_f,B_f,C_f_3,D_f_3]   = tf2ss([1, 0, 0],[1, lambda_1, lambda_0]); % <-- Fill in blanks
% A_f and B_f is only needed once as it is the same all over. Hence tildes

%% Adaptive gain
gamma   = diag([500, 100, 100]);                               % <-- Fill in blanks

%% Simulation MATLAB
% Define input as a function of t
u       = @(n) sin(t(n));                               % <-- Fill in blanks


% Memory allocation
x       = zeros(2, N);
x_z     = zeros(2, 1);
x_phi   = zeros(2, 1);
theta   = zeros(3, N);
U       = zeros(N-1,1);
masses  = zeros(1, N);

% Initial estimates
theta(:,1) = [0; 0; 0];                            % <-- Fill in blanks
masses(:,1) = m(1);

% Main loop. Simulate using forward Euler (x[k+1] = x[k] + h*x_dot)
for n = 1:N-1

    % Simulate true system
    x_dot           = A(n)*x(:, n) + B(n)*u(n);
    x(:, n+1)       = x(:, n) + h*x_dot;
    y               = x(1, n);

    % Generate z and phi by filtering known signals
    x_z_n           = x_z + (A_f*x_z + B_f*u(n))*h;   % u is unfiltered 'z'
    z               = C_f_1*x_z;                      % 1/Lambda * u

    x_phi_n         = x_phi + (A_f*x_phi + B_f*y)*h;
                    %      s^2/Lambda * y           s/Lambda * y               1/Lambda * y
    phi             = [(C_f_3*x_phi + D_f_3*y); (C_f_2*x_phi + D_f_2*y); (C_f_1*x_phi + D_f_1*y)];

    % Calculate estimation error
    epsilon         = z - theta(:, n)'*phi;                    % <-- Fill in blanks

    % Update law
    theta_dot       = gamma*epsilon*phi;     % <-- Fill in blanks
    theta(:, n+1)   = theta(:, n) + theta_dot*h;

    % Set values for next iteration
    x_phi           = x_phi_n;
    x_z             = x_z_n;
    masses(:, n+1) = m(n+1);
end

% Plots
figure
subplot(3,1,1)
plot(t, theta(1,:)); hold on
plot(t, masses(1,:)); hold off
ylabel('m')
legend('estimate','true value')
grid
subplot(3,1,2)
plot(t, theta(2,:)); hold on
plot([t(1), t(end)],[beta, beta]); hold off
ylabel('\beta')
legend('estimate','true value')
grid
subplot(3,1,3)
plot(t, theta(3,:)); hold on
plot([t(1), t(end)],[k, k]); hold off
ylabel('k')
grid
legend('estimate','true value')
title('Parameter estimates')
xlabel('t [s]')
