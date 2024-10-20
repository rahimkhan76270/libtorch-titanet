#ifndef DATASET_H
#define DATASET_H
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include "spectrogram.h"
class MelSpectrogramDataset : public torch::data::datasets::Dataset<MelSpectrogramDataset> {
private:
    std::vector<std::string> file_paths;
    MelSpectrogramParams params_;
    float duration_;

public:
    MelSpectrogramDataset(const std::vector<std::string>& paths,
                          MelSpectrogramParams& params,
                          float duration=1.0) ;
    torch::data::Example<> get(size_t index) override ;
    torch::optional<size_t> size() const override;
};

#endif