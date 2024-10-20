#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <sndfile.h>
#include "dataset.h"
#include "spectrogram.h"
#include "read_audio.h"


pair<vector<float>,int > read_audio_file(const char* filename) {
    SF_INFO sfinfo;
    SNDFILE *file = sf_open(filename, SFM_READ, &sfinfo);
    
    if (!file) {
        cerr << "Error opening file: " << sf_strerror(file) << endl;
        return {vector<float>(0),0};
    }

    std::vector<float> samples(sfinfo.frames * sfinfo.channels);
    sf_count_t num_samples_read = sf_readf_float(file, samples.data(), sfinfo.frames);
    
    if (num_samples_read < 0) {
        cerr << "Error reading samples: " << sf_strerror(file) << endl;
        sf_close(file);
        return {vector<float>(0),0};
    }
        
    sf_close(file);
    pair<vector<float>,int> result={samples,sfinfo.samplerate};
    return result;
}


MelSpectrogramDataset::MelSpectrogramDataset(const std::vector<std::string>& paths,
                                            MelSpectrogramParams& params,
                                            float duration) : file_paths(paths),params_(params),duration_(duration) {
                                            }


torch::data::Example<> MelSpectrogramDataset:: get(size_t index) {
    string path=file_paths[index];
    auto audio_sr=read_audio_file(&path[0]);
    auto it=audio_sr.first.begin();
    vector<float> waveform(it,it+static_cast<int>(this->duration_*this->params_.sample_rate));
    torch::Tensor spect=mel_spectrogram(torch::tensor(waveform),this->params_);

    return {spect,torch::Tensor()};
}
torch::optional<size_t>MelSpectrogramDataset::size()const{
        return file_paths.size();
    }
