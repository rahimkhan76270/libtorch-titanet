#ifndef SPECTROGRAM_H
#define SPECTROGRAM_H
#include<torch/torch.h>
torch::Tensor mel_filter_bank(int num_filters, int fft_size, float sample_rate, float min_hz, float max_hz);
torch::Tensor mel_spectrogram(torch::Tensor signal, int frame_length, int hop_length, int num_mel_filters, float sample_rate);
#endif