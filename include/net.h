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

class MegaBlock: public torch::nn::Module{
    public:
        MegaBlock(int64_t out_channels,int64_t kernel_size,int64_t dilation,int64_t repeat,int64_t reduction);
        torch::Tensor forward(torch::Tensor x);
    
    private:
        torch::nn::Sequential repeat_block{nullptr};
        torch::nn::Conv1d depthwise_conv1{nullptr};
        torch::nn::Conv1d depthwise_conv2{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::ReLU relu{nullptr};
        torch::nn::Conv1d pointwise_conv{nullptr};
        torch::nn::BatchNorm1d norm{nullptr};
        torch::nn::ModuleHolder<SqueezeExcitation> se_block{nullptr};

};

class Epilogue : public torch::nn::Module {
public:
    Epilogue(int64_t in_channels, int64_t out_channels, int64_t kernel_size = 1);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv1d conv{nullptr};
    torch::nn::BatchNorm1d norm{nullptr};
    torch::nn::ReLU relu{nullptr};
};

class Encoder : public torch::nn::Module {
public:
    Encoder(int64_t prolog_in_channels = 80, int64_t prolog_out_channels = 256, int64_t epilog_out_channels = 256,
            int64_t kernel_b1 = 7, int64_t dilation_b1 = 1, int64_t repeat_b1 = 2, int64_t reduction_b1 = 16,
            int64_t kernel_b2 = 11, int64_t dilation_b2 = 1, int64_t repeat_b2 = 2, int64_t reduction_b2 = 16,
            int64_t kernel_b3 = 15, int64_t dilation_b3 = 1, int64_t repeat_b3 = 2, int64_t reduction_b3 = 16);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleHolder<prologue> prolog;
    torch::nn::ModuleHolder<MegaBlock> block1, block2, block3;
    torch::nn::ModuleHolder<Epilogue> epilog;
};

class AttentiveStatisticalPooling : public torch::nn::Module {
public:
    AttentiveStatisticalPooling(int64_t in_size, int64_t hidden_size, double eps = 1e-8);

    torch::Tensor forward(torch::Tensor x);

private:
    double eps;
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Tanh tanh;
    torch::nn::Softmax softmax;
};

class Decoder : public torch::nn::Module {
public:
    Decoder(int64_t in_size, int64_t hidden_size, int64_t num_class, double eps = 1e-8);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    torch::nn::ModuleHolder<AttentiveStatisticalPooling> attention;
    torch::nn::BatchNorm1d norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
};

class TiTaNet : public torch::nn::Module {
public:
    TiTaNet(int64_t prolog_in_channels = 80,
                int64_t prolog_out_channels = 256,
                int64_t epilog_out_channels = 1536,
                int64_t kernel_b1 = 3,
                int64_t dilation_b1 = 1,
                int64_t repeat_b1 = 3,
                int64_t reduction_b1 = 16,
                int64_t kernel_b2 = 3,
                int64_t dilation_b2 = 1,
                int64_t repeat_b2 = 3,
                int64_t reduction_b2 = 16,
                int64_t kernel_b3 = 3,
                int64_t dilation_b3 = 1,
                int64_t repeat_b3 = 3,
                int64_t reduction_b3 = 16,
                int64_t hidden_size = 128,
                int64_t num_class = 100,
                double eps = 1e-8);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    torch::nn::ModuleHolder<Encoder> encoder;
    torch::nn::ModuleHolder<Decoder> decoder;
};

#endif