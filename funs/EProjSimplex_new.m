function [x, ft] = EProjSimplex_new(v, k)% A(i,idxa0) = EProjSimplex_new(ad)

%
%% Problem
%
%  min  1/2 || x - v||^2             公式23 第一个论文公式15
%  s.t. x>=0, 1'x=1
%

if nargin < 2
    k = 1;
end

ft=1;
n = length(v);           % 返回v中最大数组维度的长度

v0 = v-mean(v) + k/n;    % 集中到0范围内
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;         % 输出的是判断后的向量 posidx 是一个逻辑向量，值为 1 的位置表示 v1 中的正值元素，值为 0 的位置表示非正值元素。它将在后续用于筛选出正值元素。
        npos = sum(posidx);
        g = -npos;          %g 表示梯度方向，在拉格朗日乘子法中，g 用于调整 lambda_m，从而更新 v1
        f = sum(v1(posidx)) - k;      % v1(posidx)取元素为正的元素 看看v1与k的差距
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100 %迭代次数是否超过 100。如果超过了，就停止迭代，并直接将负值元素置为 0。
            x = max(v1,0);    % 负数不取
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end