using Statistics
using Serialization

include("filter.jl")

function compute_mu_v2(df, class, feature_size, class_index)
    class_size = length(class)
    mu_vec = zeros(Float16, 1, feature_size, class_size)
    for i = 1:class_size
        c = class[i]
        current_class_pos = (df[:, class_index] .- c) .< Float16(0.1)
        current_df = df[current_class_pos,1:class_index-1]
        current_df = Float32.(current_df)
        mu = mean(current_df, dims = 1)
        mu_vec[1,:,i] = mu
    end
    return mu_vec
end

function get_order_accuracy_column(preds, truths)
    num_columns = size(preds, 2)
    accuracy_per_column = Dict{Int, Float64}()
    
    for i in 1:num_columns
        column_accuracy = sum(preds[:, i] .== truths) / length(truths)
        accuracy_per_column[i] = column_accuracy
    end
    
    sorted_indices = sort(collect(keys(accuracy_per_column)), by=x->accuracy_per_column[x], rev=true)
    
    return sorted_indices
end

function cascade_classify(df, batch_size)
    class = unique(df[:, end])
    class_index = size(df)[2]

    col_size = size(df)[2]
    feature_size = col_size - 1

    # df_sample = better_split(df, 0.00001) 
    mu_vec = compute_mu_v2(df, class, feature_size, class_index)
    # 2nd phase loop over column vector once more to estimate feature distinctiveness 
    preds = classify_all_d1(df, mu_vec, feature_size, batch_size)
    truth = df[:, class_index]
    return (truth, preds)
end

# iterate over all instance then run classify_by_distance
function classify_all_d1(X, mu_vec, feature_size, batch_size=100)
    # prepare vector to store result
    result = zeros(Float16, size(X)[1], feature_size)
    rounding_limit = Int(floor(size(X)[1] / batch_size))
    for chunk = 1:batch_size:rounding_limit*batch_size
        vec = X[chunk:chunk+batch_size-1, 1:feature_size]
        pred = classify_by_distance_features(vec, mu_vec)
        result[chunk:chunk+batch_size-1, :] = pred
    end

    # post processing
    prev_index = rounding_limit * batch_size
    vec = X[prev_index:end, 1:feature_size]
    result[prev_index:end, :] = classify_by_distance_features(vec, mu_vec)
    return result
end

function vectorized_d1_distance(X, mu)
    # make X has depth channel of depth num_class
    numclass = size(mu)[3]
    # repeat data vector to channel depth
    X = repeat(X, outer = [1,1,numclass])
    # compute the distance
    subtracted_vector = abs.(X .- mu)
    return subtracted_vector
end

function get_most_accurate_column(preds, truths)
    num_columns = size(preds, 2)
    
    best_column = argmax([sum(preds[:, i] .== truths) for i in 1:num_columns])
    
    return best_column
end

function classify_by_distance_features(X, mu)
    # X is now passed as vector of [nrow, nfeature]
    # mu is vector of [1, nfeature, nclass] but we will transform it as [nrow, nfeature, nclass]
    num_instance = size(X)[1]
    mu_vec = repeat(mu, outer=[num_instance, 1, 1])
    dist_vec = vectorized_d1_distance(X, mu_vec)

    min_vector = argmin(dist_vec, dims=3)
    min_index = @.get_min_index(min_vector)
    return min_index
end

function vectorized_euclid_distance(X, mu)
    # make X has depth channel of depth num_class
    numclass = size(mu)[3]
    # repeat data vector to channel depth
    X = repeat(X, outer = [1,1,numclass])
    # compute the distance
    subtracted_vector = X .- mu
    power_vector = subtracted_vector .^ 2
    sum_val = sum(power_vector, dims=2)
    dist_vec = sum_val .^ (1/2)
    return dist_vec
end

function classify_by_distance_v2(X, mu)
    # X is now passed as vector of [nrow, nfeature]
    # mu is vector of [1, nfeature, nclass] but we will transform it as [nrow, nfeature, nclass]
    num_instance = size(X)[1]
    mu_vec = repeat(mu, outer = [num_instance, 1, 1])
    dist_vec = vectorized_euclid_distance(X, mu_vec)
    min_vector = argmin(dist_vec, dims=3)
    min_index = @.get_min_index(min_vector)
    return min_index
end

function classify_v2(df, batch_size)
    class = unique(df[:,end])
    class_index = size(df)[2]

    col_size = size(df)[2]
    feature_size = col_size-1
    
    # df_sample = better_split(df, 0.00001) 
    mu_vec = @time compute_mu_v2(df, class, feature_size, class_index)

    # mu_vec = @time fast_compute_mu(df, class, feature_size, class_index, batch_size)
    preds = @time classify_all_v2(df, mu_vec, feature_size, batch_size)
    df = hcat(df[:,class_index], preds)
    return df
end

function true_correctness(conf_mat)
    correctness = zeros(Float32, size(conf_mat)[1])
    for i=1:size(conf_mat)[1]
        correctness[i] = conf_mat[i,i] / sum(conf_mat[:,i])
    end
    return correctness
end

function compute_mu_col(data_matrix, class_matrix, class_type)
    means = zeros(Float16, length(class_type))
    for i in eachindex(class_type)
        indices = findall(==(class_type[i]), class_matrix)
        class_data = convert(Vector{Float32}, data_matrix[indices])
        means[i] = mean(class_data)
    end
    return means
end

function confusion_matrix(Y,P)
    num_class = length(unique(Y))
    # cast all data to int 
    Y = Int8.(Y)
    P = Int8.(P)
    # create empty matrix of size (num_class, num_class)
    cnf = zeros(Int32, num_class, num_class)
    num_instance = length(Y)
    # loop over all label
    @simd for i=1:num_instance
        @inbounds @fastmath cnf[Y[i], P[i]] += 1
    end
    return cnf
end

function measure_corretness(truths, gpreds, preds,class)
    # compute global correctness for gpreds 
    _, nc = size(preds)
    valuation = confusion_matrix(truths, gpreds)
    gcorrectness_vector = true_correctness(valuation)
    gcorrectness_vector = repeat(gcorrectness_vector', outer=[nc 1])
    
    
    # compute correctness for each feature in preds
    current_correctness_vector = zeros(nc, length(class))
    for i=1:nc
        valuation = confusion_matrix(truths, preds[:,i])
        current_correctness = true_correctness(valuation)
        current_correctness_vector[i, :] = current_correctness
    end
    #display(current_correctness_vector)
    diff = gcorrectness_vector .- current_correctness_vector
    println(sum(diff, dims=2))
end

function filter_incorrect_rows(data, best_column, truths)
    incorrect_indices = data[:, best_column] .!= truths
    filtered_data = data[incorrect_indices, [best_column, size(data, 2)]]
    # Menambahkan kolom truths ke filtered_data
    filtered_data = hcat(filtered_data, truths[incorrect_indices])
    
    return filtered_data
end

function classify_all_v2(X, mu_vec, feature_size, batch_size=100)
    # prepare vector to store result
    result = zeros(Float16, size(X)[1])
    rounding_limit = Int(floor(size(X)[1]/batch_size))
    for chunk=1:batch_size:rounding_limit*batch_size
        vec = X[chunk:chunk + batch_size - 1, 1:feature_size]
        pred = classify_by_distance_v2(vec, mu_vec)
        result[chunk: chunk + batch_size - 1] = pred
    end
    # post processing
    prev_index = rounding_limit*batch_size
    vec = X[prev_index:end, 1:feature_size]
    result[prev_index:end] = classify_by_distance_v2(vec, mu_vec)

    return result
end

function cascade_column(column_data)
    unique_classes = unique(column_data[:, end])  # Ambil kelas unik dari kolom kelas
    
    # mean = compute_mu_v2(column_data[:, 1], unique_classes, 1, column_data[:, 2])
    means = compute_mu_col(column_data[:, 1], column_data[:, 2], unique_classes)
    
    return means
end

function get_min_index(X)
    return X[3]
end

path = "data_9m.mat"

df = deserialize(path)

class = unique(df[:, end])
class_index = size(df)[2]

col_size = size(df)[2]
feature_size = col_size - 1
class_size = length(class)
num_instance = size(df)[1]
batch_size = Int(floor(num_instance / 2000))

gpreds = classify_v2(df, batch_size)

truths, preds = cascade_classify(df, batch_size)
measure_corretness(truths, gpreds[:, 2], preds, class)

order_accuracy = get_order_accuracy_column(preds, truths)

winner_index = get_most_accurate_column(preds, truths)

cascade_matrix = zeros(Float16, 3, 4)
cascade_truths = zeros(Float16, 1)
for i in order_accuracy
    column_data = filter_incorrect_rows(df, i, truths)
    cascade_mean = cascade_column(column_data)
    println("Column ", i, " Cascade mean: ")
    display(cascade_mean)
    cascade_matrix[:, i] = cascade_mean

    display(column_data[:, end])
end

display(cascade_matrix)

evaluate = confusion_matrix(truths, gpreds[:, 2])
display(evaluate)
true_correctness(evaluate)
# display(length(preds))
# display(length(gpreds[:, 2]))





# tis = filter_correct_rows(df, winner_index, truths)
# display(tis)