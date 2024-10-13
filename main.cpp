#include <iostream>
#include <torch/torch.h>
#include "prologue.h"
#include "subblock.h"
#include "SqueezeExcitation.h"

using namespace std;
int main() {
    // Create a new Net
    auto prolog = make_shared<prologue>(1,3,3);
    auto subblk = make_shared<subblock>(2,2,1);
    auto sqz=make_shared<SqueezeExcitation>(128,32);

    // Create a random input tensor
    auto input = torch::randn({2, 128, 1000});

    // Forward pass
    // auto output = prolog->forward(input);
    // auto output =subblk->forward(input);
    auto output=sqz->forward(input);
    cout << output << endl;
    return 0;
}