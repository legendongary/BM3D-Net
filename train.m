function train()

global PS GN filtN;
PS = 7;
GN = 8;
filtN = PS^2*GN-1;
sigma = 15;

w1 = sqrt(2/filtN^2) * randn(filtN);
w2 = sqrt(2/(63*filtN)) * randn(63, filtN);
weights = [w1(:); w2(:)];
low = -inf * ones(length(weights), 1);
upp = +inf * ones(length(weights), 1);

train_data = load(sprintf('./data/train-sigma-%02d.mat', sigma));
data = cell(400, 1);
for n=1:400
    data{n}.clean = gpuArray(train_data.data{n}.clean);
    data{n}.noisy = gpuArray(train_data.data{n}.noisy);
    data{n}.index = gpuArray(train_data.data{n}.index);
end
clear train_data;

fun = @(x) lgtotal(x, data);

% tic;
% l = fun(weights);
% toc;
% fprintf('Initial mode: loss is %f.\n', l);

opts.x0 = weights;
opts.m = 5;
opts.pgtol = 0;
opts.factr = 0;
opts.maxIts = 3e2+1;
opts.maxTotalIts = 3e7;
opts.printEvery = 1;

[weight, loss, info] = lbfgsb(fun, low, upp, opts);
save(sprintf('./train/model-sigma-%02d.mat', sigma), 'weight', 'loss', 'info');

end
