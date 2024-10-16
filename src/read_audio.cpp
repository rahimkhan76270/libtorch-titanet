#include "read_audio.h"
#include <sndfile.h>
#include <iostream>
using namespace std;

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
