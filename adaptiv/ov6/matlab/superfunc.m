function dydt = superfunc(t,y, m, u, beta)
    dydt = [y(2); 1/m(t) * (u(t) - beta * y(2));];
