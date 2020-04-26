function grid_image(realnvp::RealNVP, xx, yy)
    #size(xx) = size(yy) = (150, 150)
    lines = vcat(reshape(xx, (1, length(xx))), reshape(yy, (1, length(yy))))
    lines = convert(atype, lines)
    img_lines = realnvp(lines)
    img_xx, img_yy = img_lines[1,:], img_lines[2,:]
    img_xx = convert(Array{Float32}, reshape(img_xx, size(xx)))
    img_yy = convert(Array{Float32}, reshape(img_yy, size(yy)))
    return img_xx, img_yy
end

function class_logits(prior::Prior, x)
    log_probs = []
    for g in prior.gaussians
        push!(log_probs, mylogpdf(g, x))
    end
    log_probs = hcat(log_probs...) #n_instances x n_components
    log_probs_weighted = log_probs .+ log.(softmax(prior.weights))
    return log_probs_weighted
end

function classify(prior::Prior, x)
    log_probs = class_logits(prior, x)
    return [arg[2]-1 for arg in argmax(log_probs, dims=2)]
end

function get_decision_boundary(f_xx, f_yy, prior)
    f_points =  vcat(reshape(f_xx, (1, length(f_xx))), reshape(f_yy, (1, length(f_yy))))
    classes = classify(prior, f_points)
    return classes
end

nothing