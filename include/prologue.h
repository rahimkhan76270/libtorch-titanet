#ifndef PROLOGUE_H
#define PROLOGUE_H
#include<torch/torch.h>

class prologue: public torch::nn::Module{
    public:
        prologue(int64_t in_channels, int64_t out_channels,int64_t kernel_size=3);
        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Conv1d conv{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::ReLU relu{nullptr};
};
#endif