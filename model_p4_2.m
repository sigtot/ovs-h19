function dxdt = model_p4_2(t,x,A,B,gamma_1,gamma_2)
% 1-2: states
% 3-4: state predictions
% 5-8: (flattened, a11, a21, a12, a22) estimates of A
% 9-10: estimates of B

dxdt = zeros(10,1);

u = 0;                                %%% Define your input here

% Simulate true system
dxdt(1:2,1) = A*x(1:2,1) + B*u;

% Reshape estimates of A from vector to matrix
A_hat = reshape(x(5:8,1),[2 2]);

% Update law for state prediction
dxdt(3:4,1) = 0;                      %%% Insert update law for x_hat

% Update law for estimates of A and B
dA_hat = 0;                           %%% Insert update law for A_hat
dxdt(5:8,1) = dA_hat(:);

dxdt(9:10,1) = 0;                     %%% Insert update law for B_hat

end

