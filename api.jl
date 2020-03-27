function extended_nn(x,config,theta,activation)
    extended_nn_ = load_op_and_grad("./build/libExtendedNn","extended_nn", multiple=true)
    x,config_,theta = convert_to_tensor([x,config,theta], [Float64,Int64,Float64])
    u, du = extended_nn_(x,config_,theta,activation)
    n = length(x)÷config[1]
    reshape(u, (n, config[end])), du 
end