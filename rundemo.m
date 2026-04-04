clc;
clear;
close all;
addpath(genpath('funs'))
load("C:\Users\admin\Desktop\24-lcg\MVC\handwritten.mat");

V = length(X);
DATA = cell(1, V);

for v = 1:V
    data{v} = double(X{v,1});
    DATA{v} = Nomalfea(data{v});
end

labels = Y';
c = max(labels);

opts.style = 0;
opts.act_fun = "max";
opts.epsilon = 0.01;
opts.lambad = 0.001;
opts_hidnum = 2000;
opts.hidnum = [opts_hidnum, opts_hidnum];
opts.NNmaxiter = 5;
opts.selectnum = 1000;
k = 150;

view_weights = ones(1, V) / V;

rng(1234, 'twister');

for nnn = 1:5

    parfor v = 1:V
        [Data{v}.train, Data{v}.test, Data{v}.train_label, Data{v}.test_label] = ...
           selectsr(DATA{v}, labels, opts.selectnum, opts.style, c);
    end

    parfor v = 1:V
        X_train = Data{v}.train;
        [~, ~, H_last{v}, ~] = ...
            SLMGC_fun(Data{v}.train, Data{v}.train_label, DATA{v}, labels, X_train, c, k, opts);
    end

    H_fused = zeros(size(H_last{1}));
    for v = 1:V
        H_fused = H_fused + view_weights(v) * H_last{v};
    end

    [ACC, MIhat, Purity] = accncutLSC(H_fused', labels');
end

disp(['ACC: ' num2str(ACC)]);
disp(['NMI: ' num2str(MIhat)]);
disp(['Purity: ' num2str(Purity)]);