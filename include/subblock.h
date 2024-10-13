#ifndef SUBBLOCK_H
#define SUBBLOCK_H
#include <torch/torch.h>
class subblock:public torch::nn::Module{
    public:
        subblock(int64_t out_channels,int64_t kernel_size,int64_t dilation=1);
        torch::Tensor forward(torch::Tensor x);
    
    private:
        torch::nn::Conv1d depthwise_conv{nullptr};
        torch::nn::Conv1d pointwise_conv{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::Dropout dropout{nullptr};
};

#endif