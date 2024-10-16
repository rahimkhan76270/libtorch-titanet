#include <iostream>
#include <torch/torch.h>
#include<torch/extension.h>    
#include "net.h"
#include "read_audio.h"

using namespace std;
int main() {
    // // Create a new Net
    // auto titanet=make_shared<TiTaNet>();

    // // Create a random input tensor
    // auto input = torch::randn({2, 80,128});
    // auto output=titanet->forward(input);
    // cout<<get<0>(output)<<endl;
    // cout<<get<1>(output)<<endl;
    pair<vector<float>,int > samples_sample_rate=read_audio_file("/home/rahim-khan/Downloads/dev-other/LibriSpeech/dev-other/116/288045/116-288045-0001.flac");
    auto samples=torch::tensor(samples_sample_rate.first,torch::dtype(torch::kFloat32));
    int sample_rate=samples_sample_rate.second;
    cout<<sample_rate<<endl;
    cout<<samples<<endl;
    return 0;
}
