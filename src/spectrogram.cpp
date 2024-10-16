#include "spectrogram.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

pair<torch::Tensor,int> _spectrogram(torch::Tensor y,
                                    torch::Tensor S,
                                    int n_fft,
                                    int hop_length,
                                    int power,
                                    int win_length,
                                    torch::Tensor window,
                                    bool center,
                                    string pad_mode){
    if (S!=NULL){
        if(n_fft!=NULL || (n_fft/2 +1)!=S.size(-2)){
            n_fft = 2 * (S.size(-2) - 1);
        }
    }
    else{
        if(n_fft==NULL){
            cerr<<"Unable to compute spectrogram with n_fft="<<n_fft<<endl;
        }
        S=torch::abs(torch::stft())
    }
}