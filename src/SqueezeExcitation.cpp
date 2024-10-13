#include "SqueezeExcitation.h"
#include <torch/torch.h>

SqueezeExcitation::SqueezeExcitation(int64_t in_channels,int64_t reduction){
    squeeze=register_module("squeeze",torch::nn::AdaptiveAvgPool1d(1));
    linear1=register_module("linear1",torch::nn::Linear(in_channels,in_channels/reduction));
    linear2=register_module("linear2",torch::nn::Linear(in_channels/reduction,in_channels));
    relu=register_module("relu",torch::nn::ReLU());
    gate=register_module("gate",torch::nn::Sigmoid());
}

torch::Tensor SqueezeExcitation::forward(torch::Tensor x){
    auto input=x;
    x=this->squeeze(x);
    x=x.squeeze(-1);
    x=this->linear1(x);
    x=this->relu(x);
    x=this->linear2(x);
    x=this->gate(x);
    x=x.unsqueeze(-1);
    return input*x.expand_as(input);
}