// #include <torch/torch.h>
// #include <iostream>
// torch::Tensor pre_emphasis(const torch::Tensor& signal, float alpha = 0.97) {
//     return torch::cat({signal.index({0}).unsqueeze(0), signal.slice(0, 1, signal.size(0)) - alpha * signal.slice(0, 0, signal.size(0) - 1)});
// }

// torch::Tensor framing(const torch::Tensor& signal, int frame_size, int frame_step) {
//     int num_frames = (signal.size(0) - frame_size) / frame_step + 1;
//     torch::Tensor indices = torch::arange(0, frame_size).unsqueeze(0) + torch::arange(0, num_frames * frame_step, frame_step).unsqueeze(1);
//     return signal.index({indices});
// }

// torch::Tensor apply_window(const torch::Tensor& frames) {
//     int frame_size = frames.size(1);
//     torch::Tensor hamming_window = 0.54 - 0.46 * torch::cos(2 * M_PI * torch::arange(frame_size) / (frame_size - 1));
//     return frames * hamming_window;
// }

// torch::Tensor compute_fft(const torch::Tensor& frames) {
//     auto fft_result = torch::fft_rfft(frames);
//     auto power_spectrum = torch::abs(fft_result).pow(2) / frames.size(1);
//     return power_spectrum;
// }

// torch::Tensor mel_filterbank(int num_filters, int fft_size, float sample_rate) {
//     auto mel_min = 2595 * log10(1 + 0 / 700.0);
//     auto mel_max = 2595 * log10(1 + 8000 / 700.0);
//     auto mel_points = torch::linspace(mel_min, mel_max, num_filters + 2);
//     auto hz_points = 700 * (torch::pow(10, mel_points / 2595) - 1);
    
//     auto fft_bins = torch::floor((fft_size + 1) * hz_points / sample_rate);
//     torch::Tensor filterbank = torch::zeros({num_filters, fft_size / 2 + 1});
    
//     for (int i = 1; i < num_filters + 1; i++) {
//         int left = fft_bins[i - 1].item<int>();
//         int center = fft_bins[i].item<int>();
//         int right = fft_bins[i + 1].item<int>();
        
//         for (int j = left; j < center; j++) {
//             filterbank.index({i - 1, j}) = (j - fft_bins[i - 1].item<float>()) / (fft_bins[i].item<float>() - fft_bins[i - 1].item<float>());
//         }
//         for (int j = center; j < right; j++) {
//             filterbank.index({i - 1, j}) = (fft_bins[i + 1].item<float>() - j) / (fft_bins[i + 1].item<float>() - fft_bins[i].item<float>());
//         }
//     }
//     return filterbank;
// }

// // Apply Mel filterbank
// torch::Tensor apply_mel_filterbank(const torch::Tensor& power_spectrum, const torch::Tensor& mel_filterbank) {
//     return torch::matmul(power_spectrum, mel_filterbank.transpose(0, 1));
// }

// // Log scaling
// torch::Tensor log_mel(const torch::Tensor& mel_spectrum) {
//     return torch::log(mel_spectrum + 1e-10);  // Adding small epsilon for numerical stability
// }

// torch::Tensor log_mel_spectrogram(torch::Tensor signal,int frame_size,int frame_step,int num_mel_filters)
// {
//     signal = pre_emphasis(signal);
//     torch::Tensor frames = framing(signal, frame_size, frame_step);
//     frames = apply_window(frames);
//     torch::Tensor power_spectrum = compute_fft(frames);
//     torch::Tensor mel_fb = mel_filterbank(num_mel_filters, frame_size, 16000);
//     torch::Tensor mel_spectrum = apply_mel_filterbank(power_spectrum, mel_fb);
//     torch::Tensor log_mel_spectrum = log_mel(mel_spectrum);
//     return log_mel_spectrum;
// }

// int main() {
//     torch::Tensor signal = torch::rand({16000}); 
//     int frame_size = 400;
//     int frame_step = 160;
//     int num_mel_filters = 26;
//     torch::Tensor log_mel_spectrum = log_mel_spectrogram(signal,frame_size,frame_step,num_mel_filters);
//     std::cout << "Mel-Spectrogram: " << log_mel_spectrum << std::endl;
    
//     return 0;
// }

#include<torch/torch.h>
#include "spectrogram.h"
using namespace std;

int main(){
    torch::Tensor waveform=torch::rand({16000});
    auto window=torch::hann_window(400);
    // auto spectt=make_shared<MelSpectrogram>(16000,400,400,160,0.0,8000.0,0,10,window);
    // auto spec=spectt->forward(waveform);
    auto spec=spectrogram(waveform,0,window,400,160,400,2,false,true,"reflect",false);
    cout<<spec<<endl;
    return 0;
}