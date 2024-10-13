#include "subblock.h"
#include <torch/torch.h>

// namespace nn=torch::nn;

subblock::subblock(int64_t out_channels,int64_t kernel_size,int64_t dilation){
    depthwise_conv =register_module("depthwise_conv",torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels,out_channels,kernel_size).padding(kernel_size/2).dilation(dilation)));
    pointwise_conv=register_module("pointwise_conv",torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels,out_channels,1)));
    relu=register_module("relu",torch::nn::ReLU());
    norm=register_module("norm",torch::nn::BatchNorm1d(out_channels));
    dropout=register_module("dropout",torch::nn::Dropout());
}


torch::Tensor subblock::forward(torch::Tensor x){
    x=this->depthwise_conv(x);
    x=this->pointwise_conv(x);
    x=this->norm(x);
    x=this->relu(x);
    x=this->dropout(x);
    return x;
}