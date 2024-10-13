#include "prologue.h"

prologue::prologue(int64_t in_channels, int64_t out_channels, int64_t kernel_size) {
    int64_t padding = (kernel_size - 1) / 2;

    conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                                     .padding(padding)));
    norm = register_module("norm", torch::nn::BatchNorm1d(out_channels));
    relu = register_module("relu", torch::nn::ReLU());
}

torch::Tensor prologue::forward(torch::Tensor x) {
    x = conv->forward(x);
    x = norm->forward(x);
    x = relu->forward(x);
    return x;
}
