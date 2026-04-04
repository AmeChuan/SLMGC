function label = LSC(Z,k, opts)



 if (~exist('opts','var'))
    opts = [];
 end
% 
% p = 200;
% if isfield(opts,'p')
%     p = opts.p;
% end
% 
% r = 6;
% if isfield(opts,'r')
%     r = opts.r;
% end


maxIter = 100;
if isfield(opts,'maxIter')
    maxIter = opts.maxIter;
end

numRep = 10;
if isfield(opts,'numRep')
    numRep = opts.numRep;
end


% mode = 'kmeans';
% if isfield(opts,'mode')
%     mode = opts.mode;
% end
% 
% nSmp=size(data,1);
% 
% % Landmark selection
% if strcmp(mode,'kmeans')
%     kmMaxIter = 5;
%     if isfield(opts,'kmMaxIter')
%         kmMaxIter = opts.kmMaxIter;
%     end
%     kmNumRep = 1;
%     if isfield(opts,'kmNumRep')
%         kmNumRep = opts.kmNumRep;
%     end
%     [dump,marks]=litekmeans(data,p,'MaxIter',kmMaxIter,'Replicates',kmNumRep);
%     clear kmMaxIter kmNumRep
% elseif strcmp(mode,'random')
%     indSmp = randperm(nSmp);
%     marks = data(indSmp(1:p),:);
%     clear indSmp
% else
%     error('mode does not support!');
% end
% 
% % Z construction
% D = EuDist2(data,marks,0);
% 
% if isfield(opts,'sigma')
%     sigma = opts.sigma;
% else
%     sigma = mean(mean(D));
% end
%  
% dump = zeros(nSmp,r);
% idx = dump;
% for i = 1:r
%     [dump(:,i),idx(:,i)] = min(D,[],2);
%     temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
%     D(temp) = 1e100; 
% end
% 
% dump = exp(-dump/(2*sigma^2));
% sumD = sum(dump,2);
% Gsdx = bsxfun(@rdivide,dump,sumD);
% Gidx = repmat([1:nSmp]',1,r);
% Gjdx = idx;
% Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);


% Graph decomposition
feaSum = full(sqrt(sum(Z,1)));
feaSum = max(feaSum, 1e-12);
Z = Z./feaSum(ones(size(Z,1),1),:);
Z=double(Z);
U = mySVD(Z,k+1);
U(:,1) = [];

U=U./repmat(sqrt(sum(U.^2,2)),1,k);

% Final kmeans
label=litekmeans(U,k,'MaxIter',maxIter,'Replicates',numRep);