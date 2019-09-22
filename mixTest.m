x = [1 2 1 2
     3 2 3 2
     6 4 6 5];
 

w = [0.1
     0.1
     0.4
     0.4];

P1 = [0.3 0   0
       0   0.4 0
       0   0   0.3];
   
P2 = [0.7 0   0
       0   0.1 0
       0   0   0.2];
   
P3 = [0.6 0   0
       0   0.2 0
       0   0   0.2];
   
P4 = [0.5 0   0
       0   0.2 0
       0   0   0.3];
 
P = [P1; P2; P3; P4];
[xmix, Pmix] = reduceGaussMix(w, x, P);
if ~isalmost(xmix, [1.5; 2.5; 5.4], 0.01)
    disp("Error wrong xmix:")
    disp(xmix)
    disp("Expected:")
    disp([1.5; 2.5; 5.4])
end

Pmix_expected = [0.79, -0.25, -0.3; -0.25, 0.46, 0.3; -0.3, 0.3, 0.69];
if ~isalmost(Pmix, Pmix_expected, 0.01)
    disp("Error wrong Pmix:")
    disp(Pmix)
    disp("Expected:")
    disp(Pmix_expected)
end
