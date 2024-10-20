#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include "spectrogram.h"
#include "net.h"
#include "dataset.h"
using namespace std;

int main(int argc, char **argv)
{
    // torch::Tensor waveform=torch::rand({16000});

    string filename = argv[1];
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "error could not open the file" << endl;
    }
    vector<std::string> filenames;
    string line;
    while (getline(file, line))
    {
        filenames.push_back(line);
    }
    file.close();
    MelSpectrogramParams params;
    auto spect_dataset = MelSpectrogramDataset(filenames, params, 1.0);
    auto mfcc_dataset = MFCCDataset(filenames,params,80);
    auto model = make_shared<TiTaNet>();
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01));
    torch::nn::CrossEntropyLoss loss_function;
    // auto spect_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(move(spect_dataset), torch::data::DataLoaderOptions(1));
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(move(mfcc_dataset), torch::data::DataLoaderOptions(1));
    int accumulated_batch_size = 0;
    torch::Tensor accumulated_data, accumulated_targets;

    for (int epoch = 0; epoch < 2; epoch++)
    {
        model->train();

        for (auto &batch : *data_loader)
        {
            auto batch_data = batch.data()->data;
            auto batch_target = batch.data()->target;
            if (accumulated_batch_size == 0)
            {
                accumulated_data = batch_data;
                accumulated_targets = batch_target;
            }
            else
            {
                accumulated_data = torch::cat({accumulated_data, batch_data}, 0);
                accumulated_targets = torch::cat({accumulated_targets, batch_target}, 0);
            }

            accumulated_batch_size += batch_data.size(0);
            if (accumulated_batch_size >= 3)
            {
                optimizer.zero_grad();
                std::tuple<torch::Tensor, torch::Tensor> output = model->forward(accumulated_data);
                torch::Tensor logits = std::get<0>(output);
                torch::Tensor loss = loss_function(logits, accumulated_targets);
                loss.backward();
                optimizer.step();
                accumulated_batch_size = 0;
                accumulated_data = torch::Tensor();
                accumulated_targets = torch::Tensor();
                std::cout << "Epoch [" << epoch + 1 << "/" << 2 << "], Loss: " << loss.item<float>() << std::endl;
            }
        }
    }

    return 0;
}