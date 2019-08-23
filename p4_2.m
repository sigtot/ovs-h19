close all;
clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Skeleton code for Problem 4.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Fill in the blanks where needed in this section and in the function
%%% "model_p4_2"

% Define true system
A = [];               
B = [];

% Inital conditions
x0 = [];

% Update law gains
gamma_1 = [];
gamma_2 = [];

% Initial estimates
x_hat_0 = [];
A_hat_0 = [];
B_hat_0 = [];

% Experiment with initial values and gains! Hint: Avoid too high gains..

%% Simulate
t_final = 500;
tspan = [0 t_final];
y0 = [x0;x_hat_0;A_hat_0(:);B_hat_0];
[t,y] = ode45(@(t,y) model_p4_2(t,y,A,B,gamma_1,gamma_2), tspan, y0);

%% Plot results

% eps
figure(1)
subplot(2,1,1)
plot(t,y(:,1)-y(:,3))
ylabel('\epsilon_1')
grid
subplot(2,1,2)
plot(t,y(:,2)-y(:,4))
ylabel('\epsilon_2')
xlabel('t [s]')
grid

% A_hat
figure(2)
subplot(2,2,1)
plot(t,y(:,5)); hold on
plot(t,a11*ones(length(t),1)); hold off
ylabel('A_{11}')
grid
subplot(2,2,2)
plot(t,y(:,7)); hold on
plot(t,a12*ones(length(t),1)); hold off
ylabel('A_{12}')
grid
subplot(2,2,3)
plot(t,y(:,6)); hold on
plot(t,a21*ones(length(t),1)); hold off
ylabel('A_{21}')
grid
subplot(2,2,4)
plot(t,y(:,8)); hold on
plot(t,zeros(length(t),1)); hold off
ylabel('A_{22}')
grid

% B_hat
figure(3)
subplot(2,1,1)
plot(t,y(:,9)); hold on
plot(t,1*ones(length(t),1)); hold off
ylabel('B_1')
grid
subplot(2,1,2)
plot(t,y(:,10)); hold on
plot(t,b2*ones(length(t),1)); hold off
ylabel('B_2')
grid
xlabel('t [s]')
