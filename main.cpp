#include <iostream>
#include <torch/torch.h>
#include "net.h"

using namespace std;
int main() {
    // Create a new Net
    auto titanet=make_shared<TiTaNet>();

    // Create a random input tensor
    auto input = torch::randn({2, 80,128});
    auto output=titanet->forward(input);
    cout<<get<0>(output)<<endl;
    cout<<get<1>(output)<<endl;
    return 0;
}