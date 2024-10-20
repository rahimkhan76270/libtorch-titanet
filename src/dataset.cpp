#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <sndfile.h>
#include "dataset.h"
#include "spectrogram.h"
#include "read_audio.h"
using namespace std;

vector<string> split(const string &str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(str);
    while (getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

pair<vector<float>, int> read_audio_file(const char *filename)
{
    SF_INFO sfinfo;
    SNDFILE *file = sf_open(filename, SFM_READ, &sfinfo);

    if (!file)
    {
        cerr << "Error opening file: " << sf_strerror(file) << endl;
        return {vector<float>(0), 0};
    }

    std::vector<float> samples(sfinfo.frames * sfinfo.channels);
    sf_count_t num_samples_read = sf_readf_float(file, samples.data(), sfinfo.frames);

    if (num_samples_read < 0)
    {
        cerr << "Error reading samples: " << sf_strerror(file) << endl;
        sf_close(file);
        return {vector<float>(0), 0};
    }

    sf_close(file);
    pair<vector<float>, int> result = {samples, sfinfo.samplerate};
    return result;
}

MelSpectrogramDataset::MelSpectrogramDataset(const std::vector<std::string> &paths,
                                             MelSpectrogramParams &params,
                                             float duration) : file_paths(paths), params_(params), duration_(duration)
{
    for (auto str : this->file_paths)
    {
        string spk_id = split(str, '/')[7];
        this->label_set.insert(spk_id);
    }
    int i = 0;
    for (auto el : this->label_set)
    {
        this->label_dict[el] = i;
        i++;
    }
}

torch::data::Example<> MelSpectrogramDataset::get(size_t index)
{
    string path = file_paths[index];
    auto audio_sr = read_audio_file(&path[0]);
    auto it = audio_sr.first.begin();
    vector<float> waveform(it, it + static_cast<int>(this->duration_ * this->params_.sample_rate));
    torch::Tensor spect = mel_spectrogram(torch::tensor(waveform).unsqueeze(0), this->params_);
    string spk_id = split(path, '/')[7];
    torch::Tensor label = torch::tensor(this->label_dict[spk_id]).unsqueeze(0);
    // cout<<spk_id<<endl;
    return {spect, label};
}
torch::optional<size_t> MelSpectrogramDataset::size() const
{
    return file_paths.size();
}

MFCCDataset::MFCCDataset(const std::vector<std::string> &paths,
                         MelSpectrogramParams &params,
                         int n_mfcc,
                         int dct_type,
                         float top_db,
                         string norm,
                         bool log_mels,
                         float duration) : file_paths(paths),
                                           params_(params), duration_(duration),
                                           n_mfcc_(n_mfcc), dct_type_(dct_type),
                                           top_db_(top_db),norm_(norm),
                                           log_mels_(log_mels)
{
    for (auto str : this->file_paths)
    {
        string spk_id = split(str, '/')[7];
        this->label_set.insert(spk_id);
    }
    int i = 0;
    for (auto el : this->label_set)
    {
        this->label_dict[el] = i;
        i++;
    }
}

torch::data::Example<> MFCCDataset::get(size_t index)
{
    string path = file_paths[index];
    auto audio_sr = read_audio_file(&path[0]);
    auto it = audio_sr.first.begin();
    vector<float> waveform(it, it + static_cast<int>(this->duration_ * this->params_.sample_rate));
    auto tensor=torch::tensor(waveform).unsqueeze(0);
    torch::Tensor mfcc = MFCC(tensor, this->params_,this->n_mfcc_,this->dct_type_,this->top_db_,this->norm_,this->log_mels_);
    string spk_id = split(path, '/')[7];
    torch::Tensor label = torch::tensor(this->label_dict[spk_id]).unsqueeze(0);
    // cout<<spk_id<<endl;
    return {mfcc, label};
}
torch::optional<size_t> MFCCDataset::size() const
{
    return file_paths.size();
}
