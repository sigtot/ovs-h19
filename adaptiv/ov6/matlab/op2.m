%close all;
%clear;

%% Create time
h       = 0.01;  % sample time (s)
N       = 10000; % number of samples
t       = 0:h:h*(N-1);

%% Insert true system
%m = @(n) meme(t(n), 20, 20*(2-exp(-0.01*(t(n)-20))), 20);
m = @(t) 15;
beta = 0.3;
k = 2;

%% A priori knowledge
k_l = 0.1;
beta_l = 0;
beta_h = 1;
m_l = 10;
tol = 0.3;
g_beta = @(beta) 1/2 - abs(beta - 1/2);
g_k = @(k) k - k_l;
g_m = @(m) m - m_l;

dg_k = @(k) 1;
dg_m = @(m) 1;
dg_beta = @(beta) -(beta - 1/2) / abs(beta - 1/2);

g_mb = @(thet) [g_beta(thet(1)); g_m(thet(2))];
dg_mb = @(thet) [dg_beta(thet(1)); dg_m(thet(2))];


%% Filter stuff
lambda_1 = 20;
lambda_0 = 50;
Lambda = [1; lambda_1; lambda_0];

%% Define filters
[  ~,  ~,C_f_1,D_f_1]   = tf2ss([1],Lambda);
[  ~,  ~,C_f_2,D_f_2]   = tf2ss([1, 0],Lambda);
[A_f,B_f,C_f_3,D_f_3]   = tf2ss([1, 0, 0],Lambda);
% A_f and B_f is only needed once as it is the same all over. Hence tildes

%% Simulation MATLAB
% Define input as a function of t
%u       = @(n) 5*sin(2*t(n)) + 15;
u       = @(t) 100*sin(2*t) + 15;

% Memory allocation
x       = zeros(2, N);
x_z     = zeros(2, 1);
x_phi   = zeros(2, 1);
theta_k = zeros(1, N);
theta_mb   = zeros(2, N);
U       = zeros(N-1,1);
masses = arrayfun(m, t);
forget = 0.1;
x_2 = zeros(2,N);
x_1 = zeros(1,N);

% Gainssss
gamma   = diag([0.01, 20, 50]);  

% Initial estimates
theta_k(1) = k_l;
theta_mb(:, 1) = [beta_l, m_l];

%% Ode45
y0 = [1; 1];
[~, x_2] = ode45(@(t,y) superfunc(t, y, m, u, beta), t, y0);
x_1 = 1/k * arrayfun(u, t)' + x_2(:, 1);

%% Estimation loop for k
% Main loop. Simulate using forward Euler (x[k+1] = x[k] + h*x_dot)
for n = 1:N-1
    z   = u(t(n));
    phi = x_1(n) - x_2(n, 1);

    % Calculate estimation error
    epsilon = (z - theta_k(:, n)'*phi)/(1 + 0.01*(phi')*phi);

    % Update lawo
    if theta_k(n) > k_l || ((theta_k(n) - tol < k_l && k_l < theta_k(n) + tol) && -gamma(1) * z * epsilon <= 0)
        theta_k_dot  = gamma(1)*epsilon*phi;
    else
        theta_k_dot = 0;
    end
    theta_k(n+1) = theta_k(n) + theta_k_dot*h;
end

%% Estimation loop for m and beta
% Main loop. Simulate using forward Euler (x[k+1] = x[k] + h*x_dot)
project = 1;
z = 1;
x_z = [-150;8.5];
for n = 1:N-1
    % Generate z and phi by filtering known signals
    x_z_n           = x_z + (A_f*x_z + B_f*(u(t(n))))*h;   % u is unfiltered 'z'
    z               = C_f_1*x_z;                      % 1/Lambda * u

    phi             = [(C_f_2*x_phi + D_f_2*x_2(n, 1)); % s/Lambda
                       (C_f_3*x_phi + D_f_3*x_2(n, 1))]; % s^2/Lambda
    x_phi_n         = x_phi + (A_f*x_phi + B_f*x_2(n, 1))*h;
    

    % Calculate estimation error
    epsilon         = (z - theta_mb(:, n)'*phi)/(1 + 0.01*(phi')*phi);
    
    dg_n = dg_mb(theta_mb(:, n));
    % Update law
    if project
        if all(g_mb(theta_mb(:, n)) > 0) || (min(abs(g_mb(theta_mb(:, n)) < tol)) && (gamma(2:3, 2:3)*epsilon*phi)'*dg_mb(theta_mb(:, n)) <= 0)
            theta_dot = gamma(2:3, 2:3)*epsilon*phi;
        else
            % I had to flip this sign               ⌄ and i do not know why
            theta_dot = gamma(2:3, 2:3)*epsilon*phi + gamma(2:3, 2:3)*(dg_n*dg_n')/(dg_n'*gamma(2:3, 2:3)*dg_n) * gamma(2:3, 2:3)*epsilon*phi;
        end
    else
        theta_dot = gamma(2:3, 2:3)*epsilon*phi;
    end
    theta_mb(:, n+1)   = theta_mb(:, n) + theta_dot*h;

    % Set values for next iteration
    x_phi           = x_phi_n;
    x_z             = x_z_n;
    masses(:, n+1) = m(n+1);
end

%% Plots
fig1 = figure(1);
subplot(3,1,1)
plot(t, theta_mb(2,:)); hold on
plot(t, masses(1,:)); hold off
ylabel('m')
title('Parameter estimates')
legend('estimate','true value')
grid
subplot(3,1,2)
plot(t, theta_mb(1,:)); hold on
plot([t(1), t(end)],[beta, beta]); hold off
ylabel('\beta')
legend('estimate','true value')
grid
subplot(3,1,3)
plot(t, theta_k(:)); hold on
plot([t(1), t(end)],[k, k]); hold off
ylabel('k')
grid
legend('estimate','true value')
xlabel('t [s]')

%% Plot system
fig2 = figure(2);
plot(t, x_1); hold on;
plot(t, x_2(:, 1));
plot(t, x_2(:, 2));
title('System states')
legend('y1', 'y2', 'y2_{dot}')
grid
