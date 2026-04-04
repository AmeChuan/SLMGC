function DATA = Nomalfea(data)
data = double(data);
mean_data = mean(data, 1);
centered_data = data - mean_data;

[coeff, score, ~, ~, explained] = pca(centered_data);

total_explained = cumsum(explained);
desired_explained = 91;
desired_dimension_auto = find(total_explained > desired_explained, 1);

reduced_data = score(:, 1:desired_dimension_auto);
reduced_data = reduced_data';

data_min = min(reduced_data(:));
data_max = max(reduced_data(:));

if data_max > data_min
    DATA = (reduced_data - data_min) / (data_max - data_min);
else
    DATA = reduced_data;
end
end