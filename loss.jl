function mylogpdf(g,x)
    xx = convert(Array{Float32}, Knet.value(x))
    ans = Distributions.logpdf(g, xx)
    return convert(atype, ans)
end

function mygradlogpdf(g,x)
    xx = convert(Array{Float32}, Knet.value(x))
    ans = []
    for i in 1:size(xx,2)
        push!(ans, Distributions.gradlogpdf(g,xx[:,i]))
    end
    ans = hcat(ans...)
    return convert(atype, ans) 
end

@Knet.primitive mylogpdf(g,x),dy 1 reshape(dy, (1,length(dy))).*mygradlogpdf(g,x) 

struct Prior; means; n_components; d; gaussians; weights; end
#n_components: number of classes
#d: feature dimension of data points
#means: d x n_components
#gaussians: we have n_components multivariate-gaussians, each with size d
function Prior(means)
    d, n_components = size(means)
    weights = convert(atype, ones(1, n_components))
    gaussians = []
    for i in 1:n_components
        mu = means[:,i]
        sig = Matrix{Float64}(I, d, d)
        push!(gaussians, MvNormal(mu, sig))
    end
    Prior(means, n_components, d, gaussians, weights)
end

function log_prob(prior::Prior, z, labels=nothing; label_weight=1.0)
    all_log_probs = []
    for g in prior.gaussians
        push!(all_log_probs, mylogpdf(g, z))
    end
    all_log_probs = hcat(all_log_probs...) #n_instances x n_components
    mixture_log_probs = logsumexp(all_log_probs .+ log.(softmax(prior.weights)); dims=2)
    if labels == nothing
        return mixture_log_probs
    else
        #log_probs = convert(atype, zeros(size(mixture_log_probs)))
        len = size(mixture_log_probs, 1)
        int_labels = permutedims(convert(Array{Int32}, labels))
        c_mixture = convert(atype, zeros(len,1))
        mask_mixture = [index[1] for index in findall(label->label==-1, int_labels)]
        c_mixture[mask_mixture,1] .= 1
        log_probs = c_mixture .* mixture_log_probs
        for i in 1:prior.n_components
            c_all_log_probs = convert(atype, zeros(len,1))
            mask = [index[1] for index in findall(label->label==(i-1), int_labels)]
            c_all_log_probs[mask] .=  label_weight
            log_probs += (c_all_log_probs .* all_log_probs[:,i:i])
        end  
        return log_probs
    end
end

function flow_loss(z, logdet, labels, prior; k=256)
    prior_ll = log_prob(prior, z, labels)
    #I dont know why we are doing this correction
    batch_size = size(z,2)
    kk = length(z) / batch_size
    
    corrected_prior_ll = prior_ll .- log(k) * kk
    if logdet == 0
        ll = corrected_prior_ll
    else
        ll = corrected_prior_ll + permutedims(logdet)
    end
    nll = -mean(ll)
    return nll
end

function forward(realnvp, data, labels, prior; k=256)
    z = realnvp(data)
    sldj = logdet(realnvp)
    return flow_loss(z, sldj, labels, prior, k=k)
end

nothing