#pragma once
#ifndef SQUEEZEEXCITATION_H
#define SQUEEZEEXCITATION_H
#include<torch/torch.h>

class SqueezeExcitation: public torch::nn::Module{
    public:
        SqueezeExcitation(int64_t in_channels,int64_t reduction);
        torch::Tensor forward(torch::Tensor x);
    
    private:
        torch::nn::AdaptiveAvgPool1d squeeze{nullptr};
        torch::nn::Linear linear1{nullptr};
        torch::nn::Linear linear2{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::Sigmoid gate{nullptr};
};


#endif