#include<torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "spectrogram.h"
using namespace std;


torch::Tensor apply_padding(torch::Tensor waveform, int pad,string pad_mode) {
    if (pad > 0) {
        return torch::constant_pad_nd(waveform, {pad, pad}, 0);
    }
    return waveform;
}
std::pair<bool, bool> get_spec_norms(string normalized) {
    bool frame_length_norm = false;
    bool window_norm = false;

    if (normalized == "window") {
        window_norm = true;
    } else if (normalized == "frame_length") {
        frame_length_norm = true;
    } else if (normalized == "True") {
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
    bool onesided
) {
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
        true 
    );
    auto batch_dims = shape.slice(0, shape.size() - 1).vec();
    batch_dims.push_back(spec_f.size(-2)); 
    batch_dims.push_back(spec_f.size(-1));
    spec_f = spec_f.reshape(batch_dims);
    if (normalized) {
        spec_f /= window.pow(2.0).sum().sqrt();
    }
    if (power == 1.0f) {
        return torch::abs(spec_f);
    } else if (power < 0) {
        return spec_f; 
    } else {
        return torch::abs(spec_f).pow(power);
    }
}


double hz_to_mel(double freq,string mel_scale) {
    if (mel_scale != "slaney" && mel_scale != "htk") {
        throw std::invalid_argument("mel_scale should be one of \"htk\" or \"slaney\".");
    }

    if (mel_scale == "htk") {
        return 2595.0 * std::log10(1.0 + (freq / 700.0));
    }

    double f_min = 0.0;
    double f_sp = 200.0 / 3.0;
    double mels = (freq - f_min) / f_sp;
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = std::log(6.4) / 27.0;

    if (freq >= min_log_hz) {
        mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
    }

    return mels;
}

torch::Tensor mel_to_hz(const torch::Tensor& mels, string mel_scale) {
    if (mel_scale != "slaney" && mel_scale != "htk") {
        throw std::invalid_argument("mel_scale should be one of \"htk\" or \"slaney\".");
    }

    torch::Tensor freqs;
    if (mel_scale == "htk") {
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

torch::Tensor create_triangular_filterbank(const torch::Tensor& all_freqs, const torch::Tensor& f_pts) {
    torch::Tensor f_diff = f_pts.index({torch::indexing::Slice(1, torch::indexing::None)}) - f_pts.index({torch::indexing::Slice(torch::indexing::None, -1)});
    torch::Tensor slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1);
    torch::Tensor zero = torch::zeros(1, torch::kFloat32);
    torch::Tensor down_slopes = (-1.0 * slopes.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -2)})) / f_diff.index({torch::indexing::Slice(torch::indexing::None, -1)}); // (n_freqs, n_filter)
    torch::Tensor up_slopes = slopes.index({torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)}) / f_diff.index({torch::indexing::Slice(1, torch::indexing::None)}); // (n_freqs, n_filter)
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
    string mel_scale
) {
    if (norm != "slaney") {
        throw std::invalid_argument("norm must be one of None or \"slaney\"");
    }

    torch::Tensor all_freqs = torch::linspace(0, sample_rate / 2, n_freqs);
    double m_min = hz_to_mel(f_min, mel_scale);
    double m_max = hz_to_mel(f_max, mel_scale);
    torch::Tensor m_pts = torch::linspace(m_min, m_max, n_mels + 2);
    torch::Tensor f_pts = mel_to_hz(m_pts, mel_scale);
    torch::Tensor fb = create_triangular_filterbank(all_freqs, f_pts);

    if (norm == "slaney") {
        torch::Tensor enorm = 2.0 / (f_pts.index({torch::indexing::Slice(2, n_mels + 2)}) - f_pts.index({torch::indexing::Slice(torch::indexing::None, n_mels)}));
        fb *= enorm.unsqueeze(0);
    }

    if (torch::any(fb.amax() == 0.0).item<bool>()) {
        std::cerr << "At least one mel filterbank has all zero values. "
                  << "The value for `n_mels` (" << n_mels << ") may be set too high. "
                  << "Or, the value for `n_freqs` (" << n_freqs << ") may be set too low." << std::endl;
    }
    return fb;
}

MelScale::MelScale(int n_mels, int sample_rate, double f_min, double f_max, int n_stft,string norm,string mel_scale)
    : n_mels_(n_mels), sample_rate_(sample_rate), f_min_(f_min), f_max_(f_max), norm_(norm), mel_scale_(mel_scale) 
{
    if (f_min_ > f_max_) {
        throw std::invalid_argument("Require f_min: " + std::to_string(f_min_) + " <= f_max: " + std::to_string(f_max_));
    }
    fb_ = melscale_fbanks(n_stft, f_min_, f_max_, n_mels_, sample_rate_, norm_, mel_scale_);
    register_buffer("fb", fb_);
}

torch::Tensor MelScale::forward(torch::Tensor specgram) {
    torch::Tensor mel_specgram = torch::matmul(specgram.transpose(-1, -2), fb_).transpose(-1, -2);
    return mel_specgram;
}

MelSpectrogram::MelSpectrogram(
    int sample_rate,
    int n_fft,
    int win_length,
    int hop_length,
    double f_min,
    double f_max,
    int pad,
    int n_mels,
    torch::Tensor window_fn,
    float power,
    bool normalized,
    string pad_mode,
    string norm,
    string mel_scale
) : sample_rate_(sample_rate),
    n_fft_(n_fft),
    win_length_(win_length),
    hop_length_(hop_length),
    f_min_(f_min),
    f_max_(f_max),
    pad_(pad),
    n_mels_(n_mels),
    power_(power),
    normalized_(normalized),
    pad_mode_(pad_mode),
    norm_(norm),
    mel_scale_1(mel_scale),
    spectrogram_(Spectrogram(n_fft_, win_length_, hop_length_, pad_, window_fn, power_, normalized_,true, pad_mode_,true)),
    mel_scale_(MelScale(n_mels_, sample_rate_, f_min_, f_max_, n_fft_ / 2 + 1, norm_, mel_scale_1))
{
    register_module("spectrogram",spectrogram_);
    register_module("mel_scale",mel_scale_);
}

torch::Tensor MelSpectrogram::forward(const torch::Tensor waveform) {
    torch::Tensor specgram = spectrogram_->forward(waveform);
    torch::Tensor mel_specgram = mel_scale_->forward(specgram);
    return mel_specgram;
}

Spectrogram::Spectrogram(
    int n_fft,
    int win_length,
    int hop_length,
    int pad,
    torch::Tensor window_fn,
    float power,
    bool normalized,
    bool center,
    string pad_mode,
    bool onesided
) : n_fft(n_fft), 
    pad(pad), 
    power(power), 
    normalized(normalized), 
    center(center), 
    pad_mode(pad_mode), 
    onesided(onesided),
    window(window_fn)
    {}

torch::Tensor Spectrogram::forward(const torch::Tensor& waveform) {
    return spectrogram(
        waveform,
        this->pad,
        this->window,
        this->n_fft,
        this->hop_length,
        this->win_length,
        this->power,
        this->normalized,
        this->center,
        this->pad_mode,
        this->onesided
    );
}