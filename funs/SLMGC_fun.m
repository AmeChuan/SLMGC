function [knum, summ, H_last, A] = SLMGC(train, train_labels, DATA, labels, X, c, k, opts, r)

knum = 0;                   
NITER =45;               
[dim, num] = size(X);        
summ = zeros(NITER, 1);      

if nargin < 9        
    r = -1;
end
rng(1234, 'twister');

hidnum = [dim opts.hidnum num];  
for ii = 2:size(hidnum, 2)
    Weight{ii} = 0.01 * randn(hidnum(ii), hidnum(ii-1));
end

%==【Initialization】==

% 初始化相似矩阵 A
distX = L2_distance_1(X, X);  
[distX1, idx] = sort(distX, 2);  
A = zeros(num);  
rr = zeros(num, 1);  

for i = 1:num
    di = distX1(i, 2:k+2);  
    rr(i) = 0.5 * (k * di(k+1) - sum(di(1:k)));        
    id = idx(i, 2:k+2);  
    A(i, id) = (di(k+1) - di) / (k * di(k+1) - sum(di(1:k)) + eps);  
end

if r <= 0
    r = mean(rr);  
end

lambda = r;  

% 初始化特征矩阵 F
A0 = (A + A') / 2;  
D0 = diag(sum(A0));  
L0 = D0 - A0;  
[F, ~, evs] = eig1(L0, num, 0);  


F = F(:, 2:(c+1));  
F = F ./ repmat(sqrt(sum(F.^2, 2)), 1, c);  

%==【Iterative Update】==
for iter = 1:NITER
    distf = L2_distance_1(F', F');  
    distx = L2_distance_1(X, X);    
    
    A = zeros(num);
    for i = 1:num
        idxa0 = 1:num;  
        dfi = distf(i, idxa0);  
        dxi = distx(i, idxa0);  
        ad = -(dxi + lambda * dfi) / (2 * r);  
        A(i, idxa0) = EProjSimplex_new(ad);  
    end

    AA = (A + A') / 2;  
    D = diag(sum(AA));  
    L = D - AA;  

    F_old = F;  
    [F, ~, ev] = eig1(L, c, 0);  
    evs(:, iter+1) = ev;  

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > 0.00000000001
        lambda = 2 * lambda;
    elseif fn2 < 0.00000000001
        lambda = lambda / 2;  
        F = F_old;
    else
        break
    end
end

% 训练深度编码器
[Weight, ~, ~] = Learnmap(Weight, A, train, opts); 
H = actfun(Weight, DATA, opts.act_fun);  
H_last = H{end};  

end
