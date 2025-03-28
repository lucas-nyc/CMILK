clear
% Initialize variables
groundtruth = readtable('groundtruth.csv'); 
load('simulation_results.mat');
rng(9);

methods = {'CMILK', 'Mean Imputation', 'Moving Median', ...
           'Moving Mean', 'KNN Imputation', 'Linear Interpolation', ...
           'Linear Regression'};

n_combinations = 65;
data = table2array(groundtruth); 
data_test_missing_sets = cell(n_combinations, 1);

n_rows = 135; 
n_missing = 9;
num_rows = size(data, 1);
num_cols = size(data, 2);
num_iterations = 100;
imputed_data_storage_MNAR = cell(n_combinations, numel(methods));
imputed_data_storage_MCAR = cell(n_missing, numel(methods));
iteration_datasets = cell(num_iterations, 1);

new_results_mnar = table([], [], [], [], [], 'VariableNames', {'Method', 'Set_idx', 'MAE', 'RMSE', 'Time'});
new_results_MCAR = table([], [], [], [], [], 'VariableNames', {'Method', 'Set_idx', 'MAE', 'RMSE', 'Time'});


for set_idx = 1:n_combinations
    fprintf('Processing combination %d\n', set_idx);
    data_test_missing = groundtruth;
    start_idx = set_idx;
    for row_idx = 1:n_rows
        combination_idx = mod(start_idx - 1 + row_idx - 1, n_combinations) + 1;
        landmark_ids = results{combination_idx, 5};
        data_test_missing{row_idx, landmark_ids} = NaN;
    end
    data_test_missing_sets{set_idx} = data_test_missing;
end
fprintf('Done processing missing dataset...');

output_dir = 'missing_datasets/MNAR/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

missing_random_dir = 'missing_datasets/MCAR/';
csv_files = dir(fullfile(missing_random_dir, '*_percent.csv'));
dataset_missing_mcar = cell(numel(csv_files), 1);

for mcar_idx = 1:numel(csv_files)
    file_path = fullfile(missing_random_dir, csv_files(mcar_idx).name);
    dataset_missing_mcar{mcar_idx} = readtable(file_path);
    fprintf('Loaded %s\n', csv_files(mcar_idx).name);
end
fprintf('Loaded %d MCAR datasets.\n', numel(csv_files));

for set_idx = 1:height(data_test_missing_sets)
    current_missing_dataset = data_test_missing_sets{set_idx};
    modified_var_names = strcat('con_', current_missing_dataset.Properties.VariableNames);
    current_missing_dataset.Properties.VariableNames = modified_var_names;
    file_name = sprintf('missing_dataset_set_%d.csv', set_idx);
    file_path = fullfile(output_dir, file_name);
    writetable(current_missing_dataset, file_path);
    fprintf('Saved missing dataset %d/%d at %s\n', set_idx, height(data_test_missing_sets), file_path);
end

fprintf('Done initializing variables\n');

% MCAR Experiment
fprintf('Starting MCAR experiment\n');

for set_idx = 1:numel(dataset_missing_mcar)
    fprintf('Processing missing dataset %d/%d\n', set_idx, n_missing);
    modified_dataset = table2array(dataset_missing_mcar{set_idx});

    for method_idx = 1:numel(methods)
        method = methods{method_idx};
        fprintf('Processing method %s for dataset %d\n', method, set_idx);
        imputed_dataset = modified_dataset;
        imputed_datasets = cell(num_iterations, 1);
        correlation_tables_MCAR = cell(num_iterations, 1);
        coefficient_tables_MCAR = cell(num_iterations, 1);
        missing_indices_results = find(isnan(imputed_dataset));
        tic;
        switch method
            case 'CMILK'
                best_log_likelihood = -inf;
                log_likelihoods = [];
                log_likelihood_threshold = 0.00003; 
                max_iterations = 100; 
                num_iterations = min(max_iterations, num_iterations); 
                
                for iteration = 1:num_iterations
                    correlation_dataset_save = imputed_dataset;
                    correlation_k_matching_dataset_save = imputed_dataset;
                    fprintf('Iteration %d...\n', iteration);
                
                    if iteration == 1
                        current_dataset = imputed_dataset;
                    else
                        current_dataset = imputed_datasets{iteration - 1};
                    end
                
                    updated_dataset = current_dataset;
                    correlation_table = cell(num_cols, num_cols);
                    coefficient_table = cell(num_cols, num_cols);
                
                    for col1 = 1:num_cols
                        for col2 = 1:num_cols
                            if col1 ~= col2
                                valid_rows = ~isnan(updated_dataset(:, col1)) & ~isnan(updated_dataset(:, col2));
                                if sum(valid_rows) > 1
                                    correlation_table{col1, col2} = corr(updated_dataset(valid_rows, col1), ...
                                                                         updated_dataset(valid_rows, col2));
                                    coefficient_table{col1, col2} = mean(updated_dataset(valid_rows, col1) ./ ...
                                                                         updated_dataset(valid_rows, col2), 'omitnan');
                                else
                                    correlation_table{col1, col2} = NaN;
                                    coefficient_table{col1, col2} = NaN;
                                end
                            else
                                correlation_table{col1, col2} = NaN;
                                coefficient_table{col1, col2} = NaN;
                            end
                        end
                    end
                
                    if iteration == 1
                        for row = 1:num_rows
                            current_row = updated_dataset(row, :);
                            modified_row = modified_dataset(row, :);
                            missing_indices = find(isnan(modified_row));
                    
                            for missing_idx = missing_indices
                                correlations = cell2mat(correlation_table(missing_idx, :));
                                coefficients = cell2mat(coefficient_table(missing_idx, :));
                                available_indices = find(~isnan(modified_row));
                                available_correlations = correlations(available_indices);
                                available_coefficients = coefficients(available_indices);
                    
                                if ~isempty(available_correlations)
                                    [~, best_idx] = max(abs(available_correlations));
                                    best_available_idx = available_indices(best_idx);
                                    best_coefficient = available_coefficients(best_idx);
                                    predicted_value = best_coefficient * current_row(best_available_idx);
                                    correlation_dataset_save(row, missing_idx) = predicted_value;
                                    updated_dataset(row, missing_idx) = predicted_value;
                                else
                                    warning('No valid available landmarks. Unable to impute row %d, column %d.', row, missing_idx);
                                end
                            end
                        end
                    else
                        for row = 1:num_rows
                            current_row = updated_dataset(row, :);
                            modified_row = modified_dataset(row, :);
                            missing_indices = find(isnan(modified_row));
                    
                            for missing_idx = missing_indices
                                correlations = cell2mat(correlation_table(missing_idx, :));
                                coefficients = cell2mat(coefficient_table(missing_idx, :));
                                available_indices = find(~isnan(modified_row));
                                available_correlations = correlations(available_indices);
                                available_coefficients = coefficients(available_indices);
                    
                                if ~isempty(available_correlations)
                                    [~, best_idx] = max(abs(available_correlations));
                                    best_available_idx = available_indices(best_idx);
                                    best_coefficient = available_coefficients(best_idx);
                                    predicted_value = best_coefficient * current_row(best_available_idx);
                                    correlation_dataset_save(row, missing_idx) = predicted_value;
                
                                    target_column = modified_dataset(:, missing_idx);
                                    observed_values = target_column(~isnan(target_column));
                
                                    if ~isempty(observed_values)
                                        k = 30; 
                                        distances = sqrt((observed_values - predicted_value).^2);
                                        [~, closest_idx] = mink(distances, k);
                                        candidate_values = observed_values(closest_idx);
                                        mu = mean(candidate_values);
                                        sigma = std(candidate_values, 'omitnan');
                                        gaussian_likelihoods = normpdf(candidate_values, mu, sigma);
                                        [~, max_idx] = max(gaussian_likelihoods);
                                        best_candidate_value = candidate_values(max_idx);
                                        updated_dataset(row, missing_idx) = best_candidate_value;
                                    else
                                        updated_dataset(row, missing_idx) = predicted_value;
                                    end
                                else
                                    warning('No valid available landmarks. Unable to impute row %d, column %d.', row, missing_idx);
                                end
                            end
                        end
                    end

                    imputed_values = updated_dataset(isnan(modified_dataset));
                    mu = mean(imputed_values, 'omitnan');
                    sigma = std(imputed_values, 'omitnan');
                    log_likelihood = -normlike([mu, sigma], imputed_values);
                    log_likelihoods = [log_likelihoods; log_likelihood];
                    fprintf('Iteration %d: Log-likelihood = %.6f\n', iteration, log_likelihood);

                    if iteration > 1
                        log_likelihood_change = log_likelihoods(end) - log_likelihoods(end - 1);
                    
                        fprintf('Iteration %d: Log-likelihood change = %.6f\n', iteration, log_likelihood_change);
                    
                        if log_likelihood_change < 0
                            fprintf('Warning: Log-likelihood decreased! Reverting to best dataset.\n');
                            updated_dataset = imputed_datasets{iteration - 1};
                            break;
                        end
                    
                        if log_likelihoods(end) > best_log_likelihood
                            best_log_likelihood = log_likelihoods(end);
                        end
                    
                        if log_likelihood_change < log_likelihood_threshold
                            fprintf('Converged at iteration %d with log-likelihood change = %.6f\n', iteration, log_likelihood_change);
                            break;
                        end
                    end
                    imputed_datasets{iteration} = updated_dataset;
                end
                imputed_dataset = updated_dataset;

            case 'Mean Imputation'
                column_means = mean(modified_dataset, 'omitnan');
                imputed_dataset = fillmissing(modified_dataset, 'constant', column_means);
            
            case 'Moving Median'
                imputed_dataset = fillmissing(modified_dataset, 'movmedian', 19);
            
            case 'Moving Mean'
                imputed_dataset = fillmissing(modified_dataset, 'movmean', 19);
            
            case 'KNN Imputation'
                imputed_dataset = modified_dataset; 
                for row = 1:size(imputed_dataset, 1)
                    for col = 1:size(imputed_dataset, 2)
                        if isnan(imputed_dataset(row, col))
                            train_column_data = modified_dataset(:, col);

                            train_data_array = modified_dataset;
                            current_row = modified_dataset(row, :);
                            train_data_array(row, :) = NaN;
                            valid_rows = ~isnan(train_column_data) & ~isnan(train_data_array(:, col));
                            if ~any(valid_rows)
                                warning(['No valid rows for KNN imputation at row ', num2str(row), ...
                                         ', column ', num2str(col), '. Using column mean.']);
                            else
                                valid_columns = ~isnan(current_row);
                                valid_train_data = train_data_array(valid_rows, valid_columns);
                                valid_current_row = current_row(valid_columns);

                                distances = sqrt(sum((valid_train_data - valid_current_row).^2, 2));

                                [~, idx] = sort(distances); 
                                k = min(10, numel(idx));
                                nearest_neighbors = train_column_data(valid_rows);
                                nearest_neighbors = nearest_neighbors(idx(1:k));
                                if ~isempty(nearest_neighbors)
                                    imputed_dataset(row, col) = mean(nearest_neighbors, 'omitnan');
                                else
                                    warning(['Empty neighbors for row ', num2str(row), ...
                                             ', column ', num2str(col), '. Using column mean.']);
                                    imputed_dataset(row, col) = mean(train_column_data, 'omitnan');
                                end
                            end
                        end
                    end
                end
    
            case 'Linear Interpolation'
                imputed_dataset = fillmissing(modified_dataset, 'linear');
            
            case 'Linear Regression'
                imputed_dataset = modified_dataset;
                num_landmarks = size(modified_dataset, 2);
                landmark_models = cell(1, num_landmarks);
                predictor_indices = cell(1, num_landmarks);
                for miss_idx = 1:num_landmarks
                    available_indices = setdiff(1:num_landmarks, miss_idx); 
                    X_train = modified_dataset(:, available_indices);
                    y_train = modified_dataset(:, miss_idx);
                    valid_rows = ~isnan(y_train) & sum(~isnan(X_train), 2) >= 3; 
                    X_train = X_train(valid_rows, :);
                    y_train = y_train(valid_rows);
                    fprintf('Landmark %d: X_train size = %d x %d, y_train size = %d\n', ...
                            miss_idx, size(X_train,1), size(X_train,2), length(y_train));
                    if size(X_train, 1) > size(X_train, 2) && ~isempty(X_train) && ~isempty(y_train)
                        landmark_models{miss_idx} = fitlm(X_train, y_train);
                        predictor_indices{miss_idx} = available_indices;
                    else
                        fprintf('Insufficient data for Landmark %d, fallback to mean.\n', miss_idx);
                        landmark_models{miss_idx} = mean(y_train, 'omitnan');
                    end
                end
             
                for row = 1:size(modified_dataset, 1)
                    current_row = modified_dataset(row, :);
                    missing_indices = find(isnan(current_row));
                    
                    for miss_idx = missing_indices
                        if isa(landmark_models{miss_idx}, 'LinearModel')
                            model_predictors = predictor_indices{miss_idx};
                            X_test = current_row(model_predictors);
                            
                            if ~any(isnan(X_test))
                                imputed_value = predict(landmark_models{miss_idx}, X_test);
                            else
                                fprintf('Row %d, Landmark %d: NaNs in X_test, using mean instead.\n', row, miss_idx);
                                imputed_value = mean(modified_dataset(:, miss_idx), 'omitnan');
                            end
                        else
                            imputed_value = landmark_models{miss_idx};
                        end
                        imputed_dataset(row, miss_idx) = imputed_value;
                    end
                end


        end

        elapsed_time = toc;
        fprintf('Method %s completed in %.4f seconds.\n', method, elapsed_time);
        imputed_data_storage_MCAR{set_idx, method_idx} = imputed_dataset;
        mae_set = mean(abs(data(missing_indices_results) - imputed_dataset(missing_indices_results)), 'omitnan');
        rmse_set = sqrt(mean((data(missing_indices_results) - imputed_dataset(missing_indices_results)).^2, 'omitnan'));
        new_results_MCAR = [new_results_MCAR; {method, set_idx.*10, mae_set, rmse_set, elapsed_time}];
    end
end

fprintf('Done MCAR experiment\n');

% MNAR Experiment

fprintf('Starting MNAR experiment\n');

for set_idx = 1:n_combinations
    fprintf('Processing missing dataset %d/%d\n', set_idx, n_combinations);
    modified_dataset = table2array(data_test_missing_sets{set_idx}); 

    for method_idx = 1:numel(methods)
        method = methods{method_idx};
        fprintf('Processing method %s for dataset %d\n', method, set_idx);
        imputed_dataset = modified_dataset; 
        imputed_datasets = cell(num_iterations, 1);
        correlation_tables = cell(num_iterations, 1);
        coefficient_tables = cell(num_iterations, 1);
        missing_indices_results = find(isnan(imputed_dataset));

        tic;
        
        switch method
            case 'CMILK'
                mse_threshold = 1e-4;
                log_likelihoods = [];
                log_likelihood_threshold = 0.00003;
                
                for iteration = 1:num_iterations
                    correlation_dataset_save = imputed_dataset;
                    correlation_k_matching_dataset_save = imputed_dataset;
                    fprintf('Iteration %d...\n', iteration);
                    if iteration == 1
                        current_dataset = imputed_dataset;
                    else
                        current_dataset = imputed_datasets{iteration - 1};
                    end
                    updated_dataset = current_dataset;
                    correlation_table = cell(num_cols, num_cols);
                    coefficient_table = cell(num_cols, num_cols);
                
                    for col1 = 1:num_cols
                        for col2 = 1:num_cols
                            if col1 ~= col2
                                valid_rows = ~isnan(updated_dataset(:, col1)) & ~isnan(updated_dataset(:, col2));
                                if sum(valid_rows) > 1
                                    correlation_table{col1, col2} = corr(updated_dataset(valid_rows, col1), ...
                                                                         updated_dataset(valid_rows, col2));
                                    coefficient_table{col1, col2} = mean(updated_dataset(valid_rows, col1) ./ ...
                                                                         updated_dataset(valid_rows, col2), 'omitnan');
                                else
                                    correlation_table{col1, col2} = NaN;
                                    coefficient_table{col1, col2} = NaN;
                                end
                            else
                                correlation_table{col1, col2} = NaN;
                                coefficient_table{col1, col2} = NaN;
                            end
                        end
                    end
                    if iteration == 1
                        for row = 1:num_rows
                            current_row = updated_dataset(row, :);
                            modified_row = modified_dataset(row, :);
                            missing_indices = find(isnan(modified_row));
                    
                            for missing_idx = missing_indices
                                correlations = cell2mat(correlation_table(missing_idx, :));
                                coefficients = cell2mat(coefficient_table(missing_idx, :));
                                available_indices = find(~isnan(modified_row));
                                available_correlations = correlations(available_indices);
                                available_coefficients = coefficients(available_indices);
                    
                                if ~isempty(available_correlations)
                                    [~, best_idx] = max(abs(available_correlations));
                                    best_available_idx = available_indices(best_idx);
                                    best_coefficient = available_coefficients(best_idx);
                                    predicted_value = best_coefficient * current_row(best_available_idx);
                                    correlation_dataset_save(row, missing_idx) = predicted_value;
                                    updated_dataset(row, missing_idx) = predicted_value;
                                else
                                    warning('No valid available landmarks. Unable to impute row %d, column %d.', row, missing_idx);
                                end
                            end
                        end
                    else
                        for row = 1:num_rows
                            current_row = updated_dataset(row, :);
                            modified_row = modified_dataset(row, :);
                            missing_indices = find(isnan(modified_row));
                    
                            for missing_idx = missing_indices
                                correlations = cell2mat(correlation_table(missing_idx, :));
                                coefficients = cell2mat(coefficient_table(missing_idx, :));
                                available_indices = find(~isnan(modified_row));
                                available_correlations = correlations(available_indices);
                                available_coefficients = coefficients(available_indices);
                    
                                if ~isempty(available_correlations)
                                    [~, best_idx] = max(abs(available_correlations));
                                    best_available_idx = available_indices(best_idx);
                                    best_coefficient = available_coefficients(best_idx);
                                    predicted_value = best_coefficient * current_row(best_available_idx);
                                    target_column = modified_dataset(:, missing_idx);
                                    observed_values = target_column(~isnan(target_column));
                                    if ~isempty(observed_values)
                                        k = 30;
                                        distances = sqrt((observed_values - predicted_value).^2);
                                        [~, closest_idx] = mink(distances, k);
                                        candidate_values = observed_values(closest_idx);
                                        mu = mean(candidate_values);
                                        sigma = std(candidate_values, 'omitnan');
                                        gaussian_likelihoods = normpdf(candidate_values, mu, sigma);
                                        [~, max_idx] = max(gaussian_likelihoods);
                                        best_candidate_value = candidate_values(max_idx);
                                        updated_dataset(row, missing_idx) = best_candidate_value;
                                    else
                                        updated_dataset(row, missing_idx) = predicted_value;
                                    end
                                else
                                    warning('No valid available landmarks. Unable to impute row %d, column %d.', row, missing_idx);
                                end
                            end
                        end
                    end
                    imputed_values = updated_dataset(isnan(modified_dataset));
                    mu = mean(imputed_values, 'omitnan');
                    sigma = std(imputed_values, 'omitnan');
                    log_likelihood = -normlike([mu, sigma], imputed_values);
                    log_likelihoods = [log_likelihoods; log_likelihood];
                    fprintf('Iteration %d: Log-likelihood = %.6f\n', iteration, log_likelihood);

                    if iteration > 1
                        log_likelihood_change = log_likelihoods(end) - log_likelihoods(end - 1);
                    
                        fprintf('Iteration %d: Log-likelihood change = %.6f\n', iteration, log_likelihood_change);
                    
                        if log_likelihood_change < 0
                            fprintf('Warning: Log-likelihood decreased! Reverting to best dataset.\n');
                            updated_dataset = imputed_datasets{iteration - 1};
                            break;
                        end
                    
                        if log_likelihoods(end) > best_log_likelihood
                            best_log_likelihood = log_likelihoods(end);
                        end
                    
                        if log_likelihood_change < log_likelihood_threshold
                            fprintf('Converged at iteration %d with log-likelihood change = %.6f\n', iteration, log_likelihood_change);
                            break;
                        end
                    end
                    imputed_datasets{iteration} = updated_dataset;
                end

                imputed_dataset = updated_dataset;

            case 'Mean Imputation'
                column_means = mean(modified_dataset, 'omitnan');
                imputed_dataset = fillmissing(modified_dataset, 'constant', column_means);
            
            case 'Moving Median'
                imputed_dataset = fillmissing(modified_dataset, 'movmedian', 19);
            
            case 'Moving Mean'
                imputed_dataset = fillmissing(modified_dataset, 'movmean', 19);
            
            case 'KNN Imputation'
                imputed_dataset = modified_dataset;
                for row = 1:size(imputed_dataset, 1)
                    for col = 1:size(imputed_dataset, 2)
                        if isnan(imputed_dataset(row, col))
                            train_column_data = modified_dataset(:, col);
                            train_data_array = modified_dataset;
                            current_row = modified_dataset(row, :);
                            train_data_array(row, :) = NaN;
                            valid_rows = ~isnan(train_column_data) & ~isnan(train_data_array(:, col));
                            if ~any(valid_rows)
                                warning(['No valid rows for KNN imputation at row ', num2str(row), ...
                                         ', column ', num2str(col), '. Using column mean.']);
                            else
                                valid_columns = ~isnan(current_row);
                                valid_train_data = train_data_array(valid_rows, valid_columns);
                                valid_current_row = current_row(valid_columns);

                                distances = sqrt(sum((valid_train_data - valid_current_row).^2, 2));

                                [~, idx] = sort(distances); 
                                k = min(10, numel(idx)); 
                                nearest_neighbors = train_column_data(valid_rows);
                                nearest_neighbors = nearest_neighbors(idx(1:k));
                                if ~isempty(nearest_neighbors)
                                    imputed_dataset(row, col) = mean(nearest_neighbors, 'omitnan');
                                else
                                    warning(['Empty neighbors for row ', num2str(row), ...
                                             ', column ', num2str(col), '. Using column mean.']);
                                    imputed_dataset(row, col) = mean(train_column_data, 'omitnan');
                                end
                            end
                        end
                    end
                end
    
            case 'Linear Interpolation'
                imputed_dataset = fillmissing(modified_dataset, 'linear');
            
            case 'Linear Regression'
                imputed_dataset = modified_dataset;
                for row = 1:size(modified_dataset, 1)
                    current_row = modified_dataset(row, :);
                    missing_indices = find(isnan(current_row));
                    available_indices = find(~isnan(current_row));
                    for miss_idx = missing_indices
                        train_dataset = modified_dataset;
                        train_dataset(row, :) = NaN;
                        X_train = train_dataset(:, available_indices);
                        y_train = train_dataset(:, miss_idx);
                        valid_rows = ~any(isnan(X_train), 2) & ~isnan(y_train);
                        X_train = X_train(valid_rows, :);
                        y_train = y_train(valid_rows);
                        if size(X_train, 1) > size(X_train, 2) && ~isempty(X_train) && ~isempty(y_train)
                            mdl = fitlm(X_train, y_train);
                            X_test = current_row(available_indices);
                            if ~any(isnan(X_test))
                                imputed_value = predict(mdl, X_test);
                            else
                                imputed_value = mean(modified_dataset(:, miss_idx), 'omitnan');
                            end
                        else
                            imputed_value = mean(modified_dataset(:, miss_idx), 'omitnan');
                        end
                        imputed_dataset(row, miss_idx) = imputed_value;
                    end
                end
        end

        elapsed_time = toc;
        fprintf('Method %s completed in %.4f seconds.\n', method, elapsed_time);
        imputed_data_storage_MNAR{set_idx, method_idx} = imputed_dataset;
        mae_set = mean(abs(data(missing_indices_results) - imputed_dataset(missing_indices_results)), 'omitnan');
        rmse_set = sqrt(mean((data(missing_indices_results) - imputed_dataset(missing_indices_results)).^2, 'omitnan'));
        new_results_mnar = [new_results_mnar; {method, set_idx, mae_set, rmse_set, elapsed_time}];
    end
end

fprintf('Done MNAR experiment\n');


% MEAN RESULTS MAE/RMSE

gainmice_results_set = readtable('imputation_results_mnar.csv');
gainmice_results_mcar = readtable('imputation_results_mcar.csv');
new_results_mnar = [new_results_mnar; gainmice_results_set(:,2:6)];
new_results_MCAR = [new_results_MCAR; gainmice_results_mcar(:,2:6)];


unique_methods_mnar = unique(new_results_mnar.Method);
mean_metrics_mnar = table(cell(numel(unique_methods_mnar), 1), zeros(numel(unique_methods_mnar), 1), zeros(numel(unique_methods_mnar), 1), zeros(numel(unique_methods_mnar), 1),...
    'VariableNames', {'Method', 'RMSE_Mean', 'MAE_Mean', 'Time'});
unique_methods_mcar = unique(new_results_MCAR.Method);
mean_metrics_mcar = table(cell(numel(unique_methods_mcar), 1), zeros(numel(unique_methods_mcar), 1), zeros(numel(unique_methods_mcar), 1), zeros(numel(unique_methods_mcar), 1),...
    'VariableNames', {'Method', 'RMSE_Mean', 'MAE_Mean', 'Time'});
    
for i = 1:numel(unique_methods_mnar)
    method_name = unique_methods_mnar(i);
    method_data = new_results_mnar(strcmp(new_results_mnar.Method, method_name), :);
    mean_rmse = mean(method_data.RMSE);
    mean_mae = mean(method_data.MAE);
    mean_time= mean(method_data.Time);
    mean_metrics_mnar.RMSE_Mean(i) = mean_rmse;
    mean_metrics_mnar.MAE_Mean(i) = mean_mae;
    mean_metrics_mnar.Method{i} = method_name;
    mean_metrics_mnar.Time(i) = mean_time;
end
fprintf('MNAR Results:\n')
disp(mean_metrics_mnar);

for i = 1:numel(unique_methods_mcar)
    method_name = unique_methods_mcar(i);
    method_data = new_results_MCAR(strcmp(new_results_MCAR.Method, method_name), :);
    valid_rows = isfinite(method_data.RMSE) & isfinite(method_data.MAE) & isfinite(method_data.Time);
    valid_data = method_data(valid_rows, :);

    if ~isempty(valid_data)
        mean_rmse = mean(valid_data.RMSE);
        mean_mae = mean(valid_data.MAE);
        mean_time = mean(valid_data.Time);
        mean_metrics_mcar.RMSE_Mean(i) = mean_rmse;
        mean_metrics_mcar.MAE_Mean(i) = mean_mae;
        mean_metrics_mcar.Method{i} = method_name;
        mean_metrics_mcar.Time(i) = mean_time;
    else
        mean_metrics_mcar.RMSE_Mean(i) = NaN;
        mean_metrics_mcar.MAE_Mean(i) = NaN;
        mean_metrics_mcar.Method{i} = method_name;
        mean_metrics_mcar.Time(i) = NaN;
    end
end
fprintf('MCAR Results:\n')
disp(mean_metrics_mcar);
