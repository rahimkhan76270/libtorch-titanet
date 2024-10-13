#include <iostream>
#include <torch/torch.h>
#include "prologue.h"


int main() {
    // Create a new Net
    auto net = std::make_shared<prologue>(1,3,3);

    // Create a random input tensor
    auto input = torch::randn({1, 1, 28});

    // Forward pass
    auto output = net->forward(input);

    std::cout << output << std::endl;
    return 0;
}