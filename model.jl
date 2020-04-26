struct Mask; d; reverse; end

#one argument: mask 
#two argument: unmask 
function (mask::Mask)(x) 
    len = size(x, 1)
    b = convert(atype,zeros(len,1))
    d = mask.d
    if mask.reverse 
        b[d+1:end,1] .= 1
    else
        b[1:d,1] .= 1
    end
    x_id = x .* b
    x_change = x .* (1 .- b)
    return x_id, x_change
end
function (mask::Mask)(y_id, y_change)
    len = size(y_id, 1)
    b = convert(atype,zeros(len,1))
    d = mask.d
    if mask.reverse 
        b[d+1:end,1] .= 1
    else
        b[1:d,1] .= 1
    end
    return y_id .* b + y_change .* (1 .- b)
end


struct Sequential
    layers
    Sequential(layers...) = new(layers)
end
(s::Sequential)(x) = (for l in s.layers; x = l(x); end; x)

struct DenseLayer; w; b; f; end

DenseLayer(i::Int,o::Int, f=relu) = DenseLayer(param(o,i), param0(o), f)

(d::DenseLayer)(x) = d.f.(d.w * x .+ d.b)



#Coupling Layer
mutable struct CouplingLayer; st_net::Sequential; mask::Mask; logdet; end

function CouplingLayer(;in_dim::Int, hidden_dim::Int, num_layers::Int, mask::Mask)
    layers = []
    push!(layers, DenseLayer(in_dim, hidden_dim, relu))
    for layer in 1:num_layers
        push!(layers, DenseLayer(hidden_dim, hidden_dim, relu))
    end
    push!(layers, DenseLayer(hidden_dim, 2*in_dim, identity))
    st_net = Sequential(layers...)
    CouplingLayer(st_net, mask, 0.0)
end

function (cpl::CouplingLayer)(x)
    x_id, x_change, s, t = get_s_and_t(cpl, x)
#     y_change = x_change .* exp.(s) .+ t #in original code, first addition is performed, then exponentiation
    y_change = (x_change .+ t) .* exp.(s) 
    y_id = x_id
    cpl.logdet = sum(s; dims=1)
    return cpl.mask(y_id, y_change)
end
#st is a neural network, the first part of the output is used as s, second part as t
function get_s_and_t(cpl::CouplingLayer, x)
    x_id, x_change = cpl.mask(x)
    st = cpl.st_net(x_id)
    middle = (size(st)[1]+1)÷2
    s, t = st[1:middle,:], st[middle+1:end,:]
    s = tanh.(s)
    return (x_id, x_change, s, t)
end


struct RealNVP; seq::Sequential; end

function RealNVP(;in_dim::Int, hidden_dim::Int, num_coupling_layers::Int, num_hidden_layers::Int)
    coupling_layers = []
    for i in 1:num_coupling_layers
        push!(coupling_layers, CouplingLayer(;in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_hidden_layers, mask=Mask(div(in_dim,2), Bool((i+1) %2))))
    end
    seq = Sequential(coupling_layers...)
    RealNVP(seq)
end

(realnvp::RealNVP)(x) = realnvp.seq(x)


function inverse(realnvp::RealNVP, z)
    return inverse(realnvp.seq, z)
end
function inverse(seq::Sequential, z)
    for layer in reverse(seq.layers)
        z = inverse(layer, z)
    end
    return z
end
function inverse(cpl::CouplingLayer, z)
    x_id, x_change, s, t = get_s_and_t(cpl, z)
#     y_change = x_change .* exp.(s) .+ t #in original code, first addition is performed, then exponentiation
    y_change = (x_change .* exp.(s*(-1))) .- t 
    y_id = x_id
    cpl.logdet = sum(s; dims=1)
    return cpl.mask(y_id, y_change)
end

function logdet(realNVP::RealNVP)
    total_logdet = 0.0
    for cpl in realNVP.seq.layers
        total_logdet = total_logdet .+ cpl.logdet
    end
    return total_logdet
end

nothing