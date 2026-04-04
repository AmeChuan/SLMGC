function Purity = computePurity(gnd, res)
    % 计算预测结果的唯一标签
    predLidx = unique(res);
    pred_classnum = length(predLidx);
    n = length(res); % 数据点总数

    % 初始化正确分类样本数量
    correnum = 0;

    for ci = 1:pred_classnum
        % 当前聚类对应的真实标签
        incluster = gnd(res == predLidx(ci));

        % 统计每个真实标签的样本数
        inclunub = histcounts(incluster, 0.5:(max(gnd) + 0.5));

        % 取最多的真实类别作为该聚类的贡献
        correnum = correnum + max(inclunub);
    end

    % 计算 Purity
    Purity = correnum / n;
end