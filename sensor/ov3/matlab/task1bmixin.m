% 1 (1 and 2)
mus = [0, 2]; mus = reshape(mus, [1,2]);
sigmas = [1, 1]; sigmas = reshape(sigmas, [1,1,2]);
w = [1/3, 1/3]; w = w(:)/sum(w(:));
[xmix, Pmix] = reduceGaussMix(w, mus, sigmas) % [1, 2]

%2 (1 and 2)
mus = [0, 2]; mus = reshape(mus, [1,2]);
sigmas = [1, 1]; sigmas = reshape(sigmas, [1,1,2]);
w = [1/6, 4/6]; w = w(:)/sum(w(:));
[xmix, Pmix] = reduceGaussMix(w, mus, sigmas) % [1.6, 1.64]

%3 (2 and 3)
mus = [2, 4.5]; mus = reshape(mus, [1,2]);
sigmas = [1.5, 1.5]; sigmas = reshape(sigmas, [1,1,2]);
w = [1/3, 1/3]; w = w(:)/sum(w(:));
[xmix, Pmix] = reduceGaussMix(w, mus, sigmas) % [3.25, 3.0625]

%4 (2 and 3)
mus = [0, 2.5]; mus = reshape(mus, [1,2]);
sigmas = [1.5, 1.5]; sigmas = reshape(sigmas, [1,1,2]);
w = [1/3, 1/3]; w = w(:)/sum(w(:));
[xmix, Pmix] = reduceGaussMix(w, mus, sigmas) % [1.25, 3.0625]