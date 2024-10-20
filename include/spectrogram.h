#ifndef SPECTROGRAM_H
#define SPECTROGRAM_H
#include<torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <string>
using namespace std;

double hz_to_mel(double freq, const string mel_scale = "htk");
torch::Tensor mel_to_hz(const torch::Tensor& mels, string mel_scale = "htk");
torch::Tensor create_triangular_filterbank(const torch::Tensor& all_freqs, const torch::Tensor& f_pts);
torch::Tensor melscale_fbanks(
    int n_freqs,
    double f_min,
    double f_max,
    int n_mels,
    int sample_rate,
    string norm="slaney",
    string mel_scale = "htk"
);
torch::Tensor apply_padding(torch::Tensor waveform, int pad, string pad_mode);
pair<bool, bool> get_spec_norms(string normalized);

torch::Tensor spectrogram(
    torch::Tensor waveform,
    int pad,
    torch::Tensor window,
    int n_fft,
    int hop_length,
    int win_length,
    float power,
    bool normalized,
    bool center,
    string pad_mode,
    bool onesided
);

struct MelSpectrogramParams {
    int sample_rate;
    int n_fft;
    int win_length;
    int hop_length;
    double f_min;
    double f_max;
    int pad;
    int n_mels;
    torch::Tensor window;
    float power;
    bool normalized;
    bool center;
    bool onesided;
    std::string pad_mode;
    std::string norm;
    std::string mel_scale;
    MelSpectrogramParams(int sample_rate = 16000,
                         int n_fft = 400,
                         int win_length = 400,
                         int hop_length = 200,
                         double f_min = 0.0,
                         double f_max = 8000.0,
                         int pad = 0,
                         int n_mels = 80,
                         torch::Tensor window = torch::hann_window(400),
                         float power = 2.0,
                         bool normalized = false,
                         bool center = true,
                         bool onesided = true,
                         std::string pad_mode = "reflect",
                         std::string norm = "slaney",
                         std::string mel_scale = "slaney");
    };

torch::Tensor mel_spectrogram(torch::Tensor waveform,
                              int sample_rate=16000,
                              int n_fft=400,
                              int win_length=400,
                              int hop_length=200,
                              double f_min=0.0,
                              double f_max=8000.0,
                              int pad=0,
                              int n_mels=80,
                              torch::Tensor window=torch::hann_window(400),
                              float power=2.0,
                              bool normalized=false,
                              bool center=true,
                              bool onesided=true,
                              string pad_mode="reflect",
                              string norm="slaney",
                              string mel_scale="htk");

torch::Tensor mel_spectrogram(torch::Tensor waveform, const MelSpectrogramParams& params);

torch::Tensor amplitude_to_DB(
    torch::Tensor x,
    float multiplier,
    float amin,
    float db_multiplier,
    std::optional<float> top_db = std::nullopt);

torch::Tensor create_dct(int n_mfcc, int n_mels, std::optional<std::string> norm = std::nullopt);
torch::Tensor MFCC(torch::Tensor& waveform,
                    const MelSpectrogramParams& params,
                    int n_mfcc=40,
                    int dct_type=2,
                    float top_db=80,
                    string norm="ortho",
                    bool log_mels=false);

#endif