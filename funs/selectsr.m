function [train_d, test_d, train_t, test_t] = selectsr(featureMat, labelMat, selectnum, style, strata_num)

    [d, N] = size(featureMat); %#ok<ASGLU>

    if nargin < 5 || isempty(strata_num)
        strata_num = max(2, min(round(sqrt(N)), selectnum));
    end

    selectnum = min(selectnum, N);

    if size(labelMat, 2) ~= N
        error('labelMat 的列数必须与 featureMat 的样本数一致。');
    end

    if style == 0
        train_d = featureMat;
        test_d = [];
        train_t = labelMat;
        test_t = [];
        return;
    end
%无标签分层
    if style == 1
        X = featureMat';
        strata_num = min(strata_num, N);

        strata_id = kmeans(X, strata_num, ...
            'Replicates', 5, ...
            'MaxIter', 1000, ...
            'EmptyAction', 'singleton', ...
            'Display', 'off');

        counts = accumarray(strata_id, 1, [strata_num, 1]);
        take = floor(selectnum * counts / N);

        remain = selectnum - sum(take);
        if remain > 0
            frac = selectnum * counts / N - take;
            [~, order] = sort(frac, 'descend');
            for i = 1:remain
                take(order(i)) = take(order(i)) + 1;
            end
        end

        train_idx = [];
        for s = 1:strata_num
            ids = find(strata_id == s);
            ns = min(take(s), numel(ids));
            if ns > 0
                rp = randperm(numel(ids));
                train_idx = [train_idx; ids(rp(1:ns))]; %#ok<AGROW>
            end
        end

        train_idx = train_idx(:)';

        if numel(train_idx) < selectnum
            rest = setdiff(1:N, train_idx, 'stable');
            need = min(selectnum - numel(train_idx), numel(rest));
            rp = randperm(numel(rest));
            train_idx = [train_idx, rest(rp(1:need))];
        elseif numel(train_idx) > selectnum
            train_idx = train_idx(1:selectnum);
        end

        test_idx = setdiff(1:N, train_idx, 'stable');

        train_d = featureMat(:, train_idx);
        test_d = featureMat(:, test_idx);
        train_t = labelMat(:, train_idx);
        test_t = labelMat(:, test_idx);
        return;
    end

    error('style 必须为 0 或 1');
end