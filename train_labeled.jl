function train_labeled(;in_dim=768, hidden_dim=512, num_coupling_layers=7, num_hidden_layers=1, k=256, lr=1e-4, steps=5001, print_freq=1000, batch_size=80, random_seed=-1, dataset="yahoo_answers", latent_mean_random=true)
    
    
    if random_seed != -1
        Random.seed!(random_seed)
    end
    
    toy_dataset = false
    if dataset == "yahoo_answers"
        dataset_ssl = make_yahoo_answers_ssl
        n_classes = 10
    elseif dataset == "ag_news"
        dataset_ssl = make_ag_news_ssl
        n_classes = 4
    elseif dataset == "hepmass"
        dataset_ssl = make_hepmass_ssl
        n_classes = 2
    elseif dataset == "miniboone"
        dataset_ssl = make_miniboone_ssl
        n_classes = 2
    elseif dataset == "moons"
        dataset_ssl = make_moons_ssl
        n_classes = 2
        toy_dataset = true
    elseif dataset == "circles"
        dataset_ssl = make_circles_ssl
        n_classes = 2
        toy_dataset = true
    elseif dataset == "8gauss"
        dataset_ssl = make_8gauss_ssl
        n_classes = 3
        toy_dataset = true
    elseif dataset == "pinwheel"
        dataset_ssl = make_pinwheel_ssl
        n_classes = 5
        toy_dataset = true
    end
    
    if toy_dataset
        data, labels, means = dataset_ssl()
    else
        data, labels, train_gt_labels, test_data, test_labels, means = dataset_ssl()
        int_test_labels = permutedims(convert(Array{Int32}, test_labels))
    end
            
        
    if latent_mean_random
        d = in_dim
        μ=zeros(d);
        Σ = Matrix{Float64}(I, d, d);
        mvn = MvNormal(μ, Σ)
        means = rand(mvn, n_classes)
    end
    
    prior = Prior(means)

    num_unlabeled = Int(sum(labels .== -1))
    num_labeled = size(labels)[2] - num_unlabeled
    
    if batch_size == "auto"
        batch_size = num_labeled
    end
    
    realnvp = RealNVP(in_dim=in_dim, hidden_dim=hidden_dim, num_coupling_layers=num_coupling_layers, num_hidden_layers=num_hidden_layers)

    int_labels = convert(Array{Int32}, labels)

    mask_labeled = [index[2] for index in findall(label->label!=-1, int_labels)]
    labeled_data = data[:,mask_labeled]
    labeled_labels = labels[mask_labeled]

    mask_unlabeled = [index[2] for index in findall(label->label==-1, int_labels)]
    unlabeled_data = data[:, mask_unlabeled]
    unlabeled_labels = labels[mask_unlabeled]


    for p in Knet.params(realnvp)
        p.opt = Adam(;lr=lr)
    end

    for step in 1:steps
        batch_idx_l = Distributions.sample(1:num_labeled, batch_size, replace=false)
        batch_x_l, batch_y_l = labeled_data[:, batch_idx_l], labeled_labels[batch_idx_l]
        batch_x = batch_x_l
        batch_y = batch_y_l
        batch_y = reshape(batch_y, (1, size(batch_y)[1]))

        loss = @diff forward(realnvp, batch_x, batch_y, prior, k=k)

        for p in Knet.params(realnvp)
            g = Knet.grad(loss, p)
            update!(Knet.value(p), g, p.opt)
        end
        if step % print_freq == 1
            print("iter ")
            print(step)
            print(" loss: ")
            print(loss)
            println(" ")
        end
    end
    
    #if toy_dataset then returns: realnvp, prior, data, int_labels
    #elseif dataset in ["miniboone" "ag_news"] then returns: realnvp, prior, test_accuracy, train_accuracy
    #else returns realnvp, prior, test_data, test_accuracy, -1
    
    if ~toy_dataset
        println("_____________________")
        inv = realnvp(test_data)
        classes = classify(prior, inv)
        test_accuracy = sum(int_test_labels .== classes)/(length(classes))
        
        if dataset in ["miniboone" "ag_news"]
            inv = realnvp(data)
            classes = classify(prior, inv)
            int_train_labels = permutedims(convert(Array{Int32}, train_gt_labels))
            train_accuracy = sum(int_train_labels .== classes)/(length(classes))
            return realnvp, prior, test_accuracy, train_accuracy
        end
        
        return realnvp, prior, test_accuracy, -1
    
    end
    return realnvp, prior, data, int_labels
end

nothing