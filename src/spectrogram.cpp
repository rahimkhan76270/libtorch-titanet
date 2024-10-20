#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "spectrogram.h"
using namespace std;

// mel spectrogram
torch::Tensor apply_padding(torch::Tensor waveform, int pad, string pad_mode)
{
    if (pad > 0)
    {
        return torch::constant_pad_nd(waveform, {pad, pad}, 0);
    }
    return waveform;
}
std::pair<bool, bool> get_spec_norms(string normalized)
{
    bool frame_length_norm = false;
    bool window_norm = false;

    if (normalized == "window")
    {
        window_norm = true;
    }
    else if (normalized == "frame_length")
    {
        frame_length_norm = true;
    }
    else if (normalized == "True")
    {
        window_norm = true;
    }

    return {frame_length_norm, window_norm};
}

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
    bool onesided)
{
    torch::Tensor padded_waveform = apply_padding(waveform, pad, pad_mode);
    auto shape = padded_waveform.sizes();
    padded_waveform = padded_waveform.reshape({-1, shape[shape.size() - 1]});
    torch::Tensor spec_f = torch::stft(
        padded_waveform,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
        true);
    auto batch_dims = shape.slice(0, shape.size() - 1).vec();
    batch_dims.push_back(spec_f.size(-2));
    batch_dims.push_back(spec_f.size(-1));
    spec_f = spec_f.reshape(batch_dims);
    if (normalized)
    {
        spec_f /= window.pow(2.0).sum().sqrt();
    }
    if (power == 1.0f)
    {
        return torch::abs(spec_f);
    }
    else if (power < 0)
    {
        return spec_f;
    }
    else
    {
        return torch::abs(spec_f).pow(power);
    }
}

double hz_to_mel(double freq, string mel_scale)
{
    if (mel_scale != "slaney" && mel_scale != "htk")
    {
        throw std::invalid_argument("mel_scale should be one of \"htk\" or \"slaney\".");
    }

    if (mel_scale == "htk")
    {
        return 2595.0 * std::log10(1.0 + (freq / 700.0));
    }

    double f_min = 0.0;
    double f_sp = 200.0 / 3.0;
    double mels = (freq - f_min) / f_sp;
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = std::log(6.4) / 27.0;

    if (freq >= min_log_hz)
    {
        mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
    }

    return mels;
}

torch::Tensor mel_to_hz(const torch::Tensor &mels, string mel_scale)
{
    if (mel_scale != "slaney" && mel_scale != "htk")
    {
        throw std::invalid_argument("mel_scale should be one of \"htk\" or \"slaney\".");
    }

    torch::Tensor freqs;
    if (mel_scale == "htk")
    {
        freqs = 700.0 * (torch::pow(10.0, mels / 2595.0) - 1.0);
        return freqs;
    }

    // Fill in the linear scale
    double f_min = 0.0;
    double f_sp = 200.0 / 3.0;
    freqs = f_min + f_sp * mels;

    // And now the nonlinear scale
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = std::log(6.4) / 27.0;

    torch::Tensor log_t = mels >= min_log_mel;
    torch::Tensor nonlinear_freqs = min_log_hz * torch::exp(logstep * (mels - min_log_mel));
    freqs = torch::where(log_t, nonlinear_freqs, freqs);

    return freqs;
}

torch::Tensor create_triangular_filterbank(const torch::Tensor &all_freqs, const torch::Tensor &f_pts)
{
    torch::Tensor f_diff = f_pts.index({torch::indexing::Slice(1, torch::indexing::None)}) - f_pts.index({torch::indexing::Slice(torch::indexing::None, -1)});
    torch::Tensor slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1);
    torch::Tensor zero = torch::zeros(1, torch::kFloat32);
    torch::Tensor down_slopes = (-1.0 * slopes.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -2)})) / f_diff.index({torch::indexing::Slice(torch::indexing::None, -1)}); // (n_freqs, n_filter)
    torch::Tensor up_slopes = slopes.index({torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)}) / f_diff.index({torch::indexing::Slice(1, torch::indexing::None)});              // (n_freqs, n_filter)
    torch::Tensor fb = torch::max(zero, torch::min(down_slopes, up_slopes));
    return fb;
}

torch::Tensor melscale_fbanks(
    int n_freqs,
    double f_min,
    double f_max,
    int n_mels,
    int sample_rate,
    string norm,
    string mel_scale)
{
    if (norm != "slaney")
    {
        throw std::invalid_argument("norm must be one of None or \"slaney\"");
    }

    torch::Tensor all_freqs = torch::linspace(0, sample_rate / 2, n_freqs);
    double m_min = hz_to_mel(f_min, mel_scale);
    double m_max = hz_to_mel(f_max, mel_scale);
    torch::Tensor m_pts = torch::linspace(m_min, m_max, n_mels + 2);
    torch::Tensor f_pts = mel_to_hz(m_pts, mel_scale);
    torch::Tensor fb = create_triangular_filterbank(all_freqs, f_pts);

    if (norm == "slaney")
    {
        torch::Tensor enorm = 2.0 / (f_pts.index({torch::indexing::Slice(2, n_mels + 2)}) - f_pts.index({torch::indexing::Slice(torch::indexing::None, n_mels)}));
        fb *= enorm.unsqueeze(0);
    }

    if (torch::any(fb.amax() == 0.0).item<bool>())
    {
        std::cerr << "At least one mel filterbank has all zero values. "
                  << "The value for `n_mels` (" << n_mels << ") may be set too high. "
                  << "Or, the value for `n_freqs` (" << n_freqs << ") may be set too low." << std::endl;
    }
    return fb;
}

MelSpectrogramParams::MelSpectrogramParams(int sample_rate,
                                           int n_fft,
                                           int win_length,
                                           int hop_length,
                                           double f_min,
                                           double f_max,
                                           int pad,
                                           int n_mels,
                                           torch::Tensor window,
                                           float power,
                                           bool normalized,
                                           bool center,
                                           bool onesided,
                                           std::string pad_mode,
                                           std::string norm,
                                           std::string mel_scale)
    : sample_rate(sample_rate),
      n_fft(n_fft),
      win_length(win_length),
      hop_length(hop_length),
      f_min(f_min),
      f_max(f_max),
      pad(pad),
      n_mels(n_mels),
      window(window),
      power(power),
      normalized(normalized),
      center(center),
      onesided(onesided),
      pad_mode(pad_mode),
      norm(norm),
      mel_scale(mel_scale) {}

torch::Tensor mel_spectrogram(torch::Tensor waveform,
                              int sample_rate,
                              int n_fft,
                              int win_length,
                              int hop_length,
                              double f_min,
                              double f_max,
                              int pad,
                              int n_mels,
                              torch::Tensor window,
                              float power,
                              bool normalized,
                              bool center,
                              bool onesided,
                              string pad_mode,
                              string norm,
                              string mel_scale)
{
    torch::Tensor specgram = spectrogram(waveform,
                                         pad,
                                         window,
                                         n_fft,
                                         hop_length,
                                         win_length,
                                         power,
                                         normalized,
                                         center,
                                         pad_mode,
                                         onesided);
    int n_stft = n_fft / 2 + 1; // This should match n_freqs in the filterbank creation
    torch::Tensor fb_ = melscale_fbanks(n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale);
    torch::Tensor mel_specgram = torch::matmul(specgram.transpose(-1, -2), fb_).transpose(-1, -2);
    return mel_specgram;
}

torch::Tensor mel_spectrogram(torch::Tensor waveform, const MelSpectrogramParams &params)
{
    torch::Tensor specgram = spectrogram(waveform,
                                         params.pad,
                                         params.window,
                                         params.n_fft,
                                         params.hop_length,
                                         params.win_length,
                                         params.power,
                                         params.normalized,
                                         params.center,
                                         params.pad_mode,
                                         params.onesided);

    torch::Tensor fb_ = melscale_fbanks(params.n_fft / 2 + 1, params.f_min, params.f_max, params.n_mels, params.sample_rate, params.norm, params.mel_scale);
    torch::Tensor mel_specgram = torch::matmul(specgram.transpose(-1, -2), fb_).transpose(-1, -2);
    return mel_specgram;
}

// MFCC
torch::Tensor amplitude_to_DB(
    torch::Tensor x,
    float multiplier,
    float amin,
    float db_multiplier,
    std::optional<float> top_db)
{
    torch::Tensor x_db = multiplier * torch::log10(torch::clamp(x, amin));
    x_db -= multiplier * db_multiplier;
    if (top_db.has_value())
    {
        auto shape = x_db.sizes();
        int64_t packed_channels = (x_db.dim() > 2) ? shape[shape.size() - 3] : 1;
        x_db = x_db.view({-1, packed_channels, shape[shape.size() - 2], shape[shape.size() - 1]});
        auto max_values = x_db.amax({-3, -2, -1}, true);
        torch::Tensor threshold = (max_values - top_db.value()).view({-1, 1, 1, 1});
        x_db = torch::max(x_db, threshold);
        x_db = x_db.view(shape);
    }

    return x_db;
}
torch::Tensor create_dct(int n_mfcc, int n_mels, std::optional<std::string> norm)
{
    if (norm.has_value() && norm.value() != "ortho")
    {
        throw std::invalid_argument("norm must be either 'ortho' or None");
    }
    torch::Tensor n = torch::arange(static_cast<float>(n_mels));
    torch::Tensor k = torch::arange(static_cast<float>(n_mfcc)).unsqueeze(1);
    torch::Tensor dct = torch::cos(M_PI / static_cast<float>(n_mels) * (n + 0.5) * k);

    if (!norm.has_value())
    {
        dct *= 2.0;
    }
    else if (norm.value() == "ortho")
    {
        dct[0] *= 1.0 / std::sqrt(2.0);
        dct *= std::sqrt(2.0 / static_cast<float>(n_mels));
    }
    return dct.t();
}

torch::Tensor MFCC(torch::Tensor& waveform,
                    const MelSpectrogramParams& params,
                    int n_mfcc,
                    int dct_type,
                    float top_db,
                    string norm,
                    bool log_mels)
{
    torch::Tensor dct=create_dct(n_mfcc,params.n_mels,norm);
    torch::Tensor mel_specgram=mel_spectrogram(waveform,params);
    if(log_mels){
        mel_specgram=torch::log(mel_specgram+1e-6);
    }
    else{
        mel_specgram=amplitude_to_DB(mel_specgram,10.0,1e-10,0,top_db);
    }
    torch::Tensor mfcc = torch::matmul(mel_specgram.transpose(-1, -2), dct).transpose(-1, -2);
    return mfcc;
}

// alternative for log mel spectrogram
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