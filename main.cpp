#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include "spectrogram.h"
// #include "read_audio.h"
#include "dataset.h"
using namespace std;

int main(int argc,char** argv){
    // torch::Tensor waveform=torch::rand({16000});
    
    string filename=argv[1];
    ifstream file(filename);
    if(!file.is_open()){
        cerr<<"error could not open the file"<<endl;
    }
    vector<std::string> filenames;
    string line;
    while (getline(file,line))
    {
        filenames.push_back(line);
    }
    file.close();
    MelSpectrogramParams params;
    auto dataset=MelSpectrogramDataset(filenames,params,1.0);
    // for(auto str:filenames){
    //     auto audio_sr=read_audio_file(&str[0]);
    //     auto waveform=torch::tensor(audio_sr.first);
    //     MelSpectrogramParams params;
    //     params.sample_rate=audio_sr.second;
    //     params.center=true;
    //     params.hop_length=60;
    //     params.win_length=500;
    //     params.n_mels=5;
    //     params.n_fft=2048;
    //     params.window=torch::hann_window(500);
    //     auto spec=mel_spectrogram(waveform,params);
    //     cout<<spec.sizes()<<endl;
    // }
    auto data_loader=torch::data::make_data_loader(dataset,torch::data::DataLoaderOptions().batch_size(3));
    for(auto& batch:*data_loader){
        auto data=batch.data();
        cout<<data->data.sizes()<<endl;
        break;
    }
    
    return 0;
}