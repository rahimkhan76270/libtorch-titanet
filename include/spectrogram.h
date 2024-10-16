#ifndef SPECTROGRAM_H
#define SPECTROGRAM_H
#include<torch/torch.h>
using namespace std;
pair<torch::Tensor,int> _spectrogram(torch::Tensor y,
                                    torch::Tensor S=NULL,
                                    int n_fft=2048,
                                    int hop_length=512,
                                    int power=1,
                                    int win_length,
                                    torch::Tensor window,
                                    bool center=true,
                                    string pad_mode="constant");
torch::Tensor hz_to_mel(torch::Tensor frequencies,bool htk=false);
torch::Tensor mel_to_hz(torch::Tensor mels,bool htk=false);
torch::Tensor mel_frequencies(int n_mels=128,float fmin=0,float fmax=11025,bool htk=false);
torch::Tensor fft_frequencies(float sr=22050,int n_fft=2048);
torch::Tensor normalize(torch::Tensor S,float norm=2,int axis=0,float threshol=NULL,bool fill=NULL);
torch::Tensor mel(float sr,int n_fft=128,float fmin=0,float fmax=NULL,bool htk=false,string norm="slanely",torch::dtype dtype=torch::kFloat32);
#endif