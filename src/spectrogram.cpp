#include "spectrogram.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>

torch::Tensor mel_filter_bank(int num_filters, int fft_size, float sample_rate, float min_hz, float max_hz) {
    int mel_points = num_filters + 2;
    torch::Tensor mel_filters = torch::empty({num_filters, fft_size});
    auto mel_min = 2595 * std::log10(1 + min_hz / 700);
    auto mel_max = 2595 * std::log10(1 + max_hz / 700);
    auto mel_bins = torch::linspace(mel_min, mel_max, mel_points);

    auto hz_bins = 700 * (torch::pow(10, mel_bins / 2595) - 1);
    for (int i = 0; i < num_filters; ++i) {
        float lower_freq = hz_bins[i].item<float>();
        float center_freq = hz_bins[i + 1].item<float>();
        float upper_freq = hz_bins[i + 2].item<float>();

        for (int j = 0; j < fft_size; ++j) {
            float freq = j * sample_rate / fft_size;
            if (freq < lower_freq || freq > upper_freq) {
                mel_filters[i][j] = 0.0;
            } else if (freq < center_freq) {
                mel_filters[i][j] = (freq - lower_freq) / (center_freq - lower_freq);
            } else if (freq < upper_freq) {
                mel_filters[i][j] = (upper_freq - freq) / (upper_freq - center_freq);
            }
        }
    }

    return mel_filters;
}

torch::Tensor mel_spectrogram(torch::Tensor signal, int frame_length, int hop_length, int num_mel_filters, float sample_rate) {
    auto stft_result = stft(signal, frame_length, hop_length);
    auto magnitude = torch::abs(stft_result);
    auto mel_filters = mel_filter_bank(num_mel_filters, frame_length, sample_rate, 0.0, sample_rate / 2.0);
    auto mel_spectrogram = torch::matmul(magnitude, mel_filters.transpose(0, 1));
    mel_spectrogram = 10 * mel_spectrogram.log10();
    
    return mel_spectrogram;
}
