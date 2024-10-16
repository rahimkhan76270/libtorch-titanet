#ifndef READ_AUDIO_H
#define READ_AUDIO_H
#include<vector>
using namespace std;
pair<vector<float>,int > read_audio_file(const char* filename);
#endif