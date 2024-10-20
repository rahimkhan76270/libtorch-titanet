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


class Spectrogram : public torch::nn::Module {
public:
    Spectrogram(
        int n_fft = 400,
        int win_length = 400,
        int hop_length = 160,
        int pad = 0,
        torch::Tensor window_fn = torch::Tensor(),
        float power = 2.0,
        bool normalized = false,
        bool center = true,
        string pad_mode = "reflect",
        bool onesided = true
    );

    torch::Tensor forward(const torch::Tensor& waveform);

private:
    int n_fft;
    int win_length;
    int hop_length;
    int pad;
    torch::Tensor window;
    float power;
    bool normalized;
    bool center;
    string pad_mode;
    bool onesided;
};

class MelScale : public torch::nn::Module{
public:
    MelScale(int n_mels, int sample_rate, double f_min, double f_max, int n_stft,string norm = "slaney",string mel_scale = "htk");
    torch::Tensor forward(const torch::Tensor specgram);

private:
    int n_mels_;
    int sample_rate_;
    double f_min_;
    double f_max_;
    std::string norm_;
    std::string mel_scale_;
    torch::Tensor fb_;
};

class MelSpectrogram : public torch::nn::Module {
public:
    MelSpectrogram(
        int sample_rate = 16000,
        int n_fft = 400,
        int win_length = 400,
        int hop_length = 160,
        double f_min = 0.0,
        double f_max = 8000.0,
        int pad = 0,
        int n_mels = 128,
        torch::Tensor window_fn = torch::hann_window(400),
        float power = 2.0,
        bool normalized = false,
        std::string pad_mode = "reflect",
        std::string norm = "slaney",
        std::string mel_scale = "htk"
    );

    torch::Tensor forward(torch::Tensor waveform);

private:
    int sample_rate_;
    int n_fft_;
    int win_length_;
    int hop_length_;
    double f_min_;
    double f_max_;
    int pad_;
    int n_mels_;
    float power_;
    bool normalized_;
    std::string pad_mode_;
    std::string norm_;
    std::string mel_scale_1;

    torch::nn::ModuleHolder<Spectrogram> spectrogram_{nullptr};
    torch::nn::ModuleHolder<MelScale> mel_scale_{nullptr};
};
#endif