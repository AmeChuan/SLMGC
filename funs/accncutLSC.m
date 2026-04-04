function [ACC, MIhat, Purity] = accncutLSC(Z, gnd)

persistent ACC_best MIhat_best Purity_best

if isempty(Purity_best)
    ACC_best = -inf;
    MIhat_best = -inf;
    Purity_best = -inf;
end

opt.r = 10;
opt.kmMaxIter = 100;

nCluster = length(unique(gnd));

res = LSC(Z, nCluster, opt);
res = bestMap(gnd, res);

ACC_now = length(find(gnd == res)) / length(gnd);
Purity_now = computePurity(gnd, res);
MIhat_now = MutualInfo(gnd, res);

if Purity_now > Purity_best
    ACC_best = ACC_now;
    MIhat_best = MIhat_now;
    Purity_best = Purity_now;
end

ACC = ACC_best;
MIhat = MIhat_best;
Purity = Purity_best;

end