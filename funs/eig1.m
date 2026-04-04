function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
                                           % 求特征值和特征向量
if nargin < 2
    c = size(A,1);%isMax：是否提取最大的特征值（1 表示提取最大，0 表示提取最小）。isSym：是否假设矩阵 A 为对称矩阵（1 表示对称，0 表示非对称）。
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end
%表示函数调用时实际传入的参数个数。如果未传入 c（即参数个数少于 2），则默认 c 为矩阵 A 的行数，也就是提取所有特征值。如果传入的 c 大于 A 的行数，也将其限制为矩阵的大小。
if nargin < 3
    isMax = 1;
    isSym = 1;
end

if nargin < 4
    isSym = 1;
end

if isSym == 1
    A = max(A,A');   % 如果是对称矩阵返回从A或A'中提取的最大元素的数组
end%v 是特征向量矩阵，d 是对角矩阵，其中对角线上的元素是 A 的特征值。
[v, d] = eig(A);   % 求矩阵A的全部特征值，构成对角阵d，并求A的特征向量构成V的列向量
d = diag(d); 
%d = real(d);
if isMax == 0
    [~, idx] = sort(d);       %  d1是排序结果，idx是相应的索引
else
    [~, idx] = sort(d,'descend');
end

idx1 = idx(1:c);    %选择排序后前 c 个特征值的索引
eigval = d(idx1);   % 使特征值组成的向量有序，提取前c个特征值
eigvec = v(:,idx1);   % 使特征向量对应于有序的特征值，提取前c个特征向量

eigval_full = d(idx);   % 全部的特征值有序，并返回