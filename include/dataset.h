#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <iostream>
#include "spectrogram.h"
using namespace std;
class MelSpectrogramDataset : public torch::data::datasets::Dataset<MelSpectrogramDataset>
{
private:
    std::vector<std::string> file_paths;
    MelSpectrogramParams params_;
    float duration_;
    set<string> label_set;
    unordered_map<string, int> label_dict;

public:
    MelSpectrogramDataset(const std::vector<std::string> &paths,
                          MelSpectrogramParams &params,
                          float duration = 1.0);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

class MFCCDataset : public torch::data::datasets::Dataset<MFCCDataset>
{
private:
    std::vector<std::string> file_paths;
    MelSpectrogramParams params_;
    float duration_;
    int n_mfcc_;
    int dct_type_;
    float top_db_;
    string norm_;
    bool log_mels_;
    set<string> label_set;
    unordered_map<string, int> label_dict;

public:
    MFCCDataset(const std::vector<std::string> &paths,
                MelSpectrogramParams &params,
                int n_mfcc = 40,
                int dct_type = 2,
                float top_db = 80,
                string norm = "ortho",
                bool log_mels = false ,
                float duration = 1.0);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

#endif