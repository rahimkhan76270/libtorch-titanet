#include "net.h"
#include<torch/torch.h>
#include<iostream>
using namespace std;
//prologue
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
//subblock
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

//squeezeExcitation
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

//mega block

MegaBlock::MegaBlock(int64_t out_channels, int64_t kernel_size, int64_t dilation, int64_t repeat, int64_t reduction) : se_block(SqueezeExcitation(out_channels, reduction)) {
    repeat_block = register_module("repeat_block", torch::nn::Sequential());
    register_module("se_block",se_block);
    for(int i = 0; i < repeat; i++){
        repeat_block->push_back(make_shared<subblock>(out_channels, kernel_size, dilation));
    }
    depthwise_conv1 = register_module("depthwise_conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel_size).padding(kernel_size/2)));
    depthwise_conv2 = register_module("depthwise_conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, kernel_size).padding(kernel_size/2)));
    dropout = register_module("dropout", torch::nn::Dropout());
    relu = register_module("relu", torch::nn::ReLU());
    pointwise_conv = register_module("pointwise_conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(out_channels, out_channels, 1)));
    norm = register_module("norm", torch::nn::BatchNorm1d(out_channels));
}


torch::Tensor MegaBlock::forward(torch::Tensor x){
    torch::Tensor y=this->repeat_block->forward(x);
    y=this->depthwise_conv1->forward(y);
    y=this->depthwise_conv2->forward(y);
    y=this->se_block->forward(x);
    x=this->pointwise_conv(x);
    x=this->norm(x);
    torch::Tensor result=this->relu(x+y);
    result=this->dropout(result);
    return result;
}

// epilogue

Epilogue::Epilogue(int64_t in_channels, int64_t out_channels, int64_t kernel_size) {
    conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)));
    norm = register_module("norm", torch::nn::BatchNorm1d(out_channels));
    relu = register_module("relu", torch::nn::ReLU());
}

torch::Tensor Epilogue::forward(torch::Tensor x) {
    x = conv->forward(x);
    x = norm->forward(x);
    x = relu->forward(x);
    return x;
}

//encoder

Encoder::Encoder(int64_t prolog_in_channels, int64_t prolog_out_channels, int64_t epilog_out_channels,
                 int64_t kernel_b1, int64_t dilation_b1, int64_t repeat_b1, int64_t reduction_b1,
                 int64_t kernel_b2, int64_t dilation_b2, int64_t repeat_b2, int64_t reduction_b2,
                 int64_t kernel_b3, int64_t dilation_b3, int64_t repeat_b3, int64_t reduction_b3)
    : prolog(prologue(prolog_in_channels, prolog_out_channels)),
      block1(MegaBlock(prolog_out_channels, kernel_b1, dilation_b1, repeat_b1, reduction_b1)),
      block2(MegaBlock(prolog_out_channels, kernel_b2, dilation_b2, repeat_b2, reduction_b2)),
      block3(MegaBlock(prolog_out_channels, kernel_b3, dilation_b3, repeat_b3, reduction_b3)),
      epilog(Epilogue(prolog_out_channels, epilog_out_channels)) {

    // Register submodules
    register_module("prolog", prolog);
    register_module("block1", block1);
    register_module("block2", block2);
    register_module("block3", block3);
    register_module("epilog", epilog);
}

// Forward pass
torch::Tensor Encoder::forward(torch::Tensor x) {
    x = prolog->forward(x);
    x = block1->forward(x);
    x = block2->forward(x);
    x = block3->forward(x);
    x = epilog->forward(x);
    return x;
}

// pooling
AttentiveStatisticalPooling::AttentiveStatisticalPooling(int64_t in_size, int64_t hidden_size, double eps)
    : eps(eps),
      linear1(register_module("linear1", torch::nn::Linear(in_size, hidden_size))),
      linear2(register_module("linear2", torch::nn::Linear(hidden_size, in_size))),
      tanh(register_module("tanh", torch::nn::Tanh())),
      softmax(register_module("softmax", torch::nn::Softmax(2))) {}

torch::Tensor AttentiveStatisticalPooling::forward(torch::Tensor x) {
    torch::Tensor input = x.clone();
    x = linear1(x.transpose(1, 2));
    x = tanh(x);
    torch::Tensor e_t = linear2(x);
    torch::Tensor alpha_t = softmax(e_t.transpose(1, 2));
    torch::Tensor means = torch::sum(alpha_t * input, 2);
    torch::Tensor residuals = torch::sum(alpha_t * input.pow(2), 2) - means.pow(2);
    torch::Tensor stds = torch::sqrt(residuals.clamp_min(eps));
    return torch::cat({means, stds}, 1);
}

//decoder
Decoder::Decoder(int64_t in_size, int64_t hidden_size, int64_t num_class, double eps)
    : attention(AttentiveStatisticalPooling( in_size, hidden_size, eps)){
        norm1=register_module("norm1", torch::nn::BatchNorm1d(in_size * 2));
      linear1=register_module("linear1", torch::nn::Linear(in_size * 2, 192));
      norm2=register_module("norm2", torch::nn::BatchNorm1d(192));
      linear2=register_module("linear2", torch::nn::Linear(192, num_class));
      register_module("attention",attention);
    }

// Forward pass
std::tuple<torch::Tensor, torch::Tensor> Decoder::forward(torch::Tensor x) {
    x = attention->forward(x);
    x = norm1(x);
    x = linear1(x);
    torch::Tensor embeddings = norm2(x);
    torch::Tensor logits = linear2(embeddings);
    return std::make_tuple(logits, embeddings);
}

//titanet

TiTaNet::TiTaNet(int64_t prolog_in_channels,
                         int64_t prolog_out_channels,
                         int64_t epilog_out_channels,
                         int64_t kernel_b1,
                         int64_t dilation_b1,
                         int64_t repeat_b1,
                         int64_t reduction_b1,
                         int64_t kernel_b2,
                         int64_t dilation_b2,
                         int64_t repeat_b2,
                         int64_t reduction_b2,
                         int64_t kernel_b3,
                         int64_t dilation_b3,
                         int64_t repeat_b3,
                         int64_t reduction_b3,
                         int64_t hidden_size,
                         int64_t num_class,
                         double eps)
    : encoder(Encoder(prolog_in_channels, 
                prolog_out_channels, 
                epilog_out_channels, 
                kernel_b1, 
                dilation_b1, 
                repeat_b1, 
                reduction_b1, 
                kernel_b2, 
                dilation_b2, 
                repeat_b2, 
                reduction_b2, 
                kernel_b3, 
                dilation_b3, 
                repeat_b3, 
                reduction_b3)),
      decoder(Decoder(epilog_out_channels, hidden_size, num_class, eps)) {
        register_module("encoder",encoder);
        register_module("decoder",decoder);
      }

std::tuple<torch::Tensor, torch::Tensor> TiTaNet::forward(torch::Tensor x) {
    x = encoder->forward(x);
    return decoder->forward(x);
}