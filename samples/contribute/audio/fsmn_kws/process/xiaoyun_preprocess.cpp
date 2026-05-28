/*
 * Copyright (c) ModelZoo. 2026-2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xiaoyun_preprocess.h"
#include "log.h"
#include "utils.h"

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace Infer {
namespace XiaoYunPreprocess {

struct WavFrontendInput {
    const float* waveform;
    size_t numSamples;
};

struct WavFrontendOutput {
    float* data;
    size_t maxFrames;
    size_t featDim;
    size_t numFrames = 0;
};

struct LfrConfig {
    int m;
    int n;
};

constexpr int SAMPLE_RATE = 16000;
constexpr int BITS_PER_SAMPLE = 16;
constexpr float LOW_FREQ = 20.0f;
constexpr int PCM_BUFFER_SIZE = 4096;
constexpr int DEFAULT_LFR_M = 5;
constexpr int DEFAULT_LFR_N = 3;
constexpr int DEFAULT_N_MELS = 80;
constexpr int DEFAULT_FRAME_LENGTH_MS = 25;
constexpr int DEFAULT_FRAME_SHIFT_MS = 10;
constexpr float DEFAULT_PREEMPH_COEFF = 0.97f;
constexpr int WAV_CHUNK_ID_SIZE = 4;

static std::vector<float> g_cmvnShift;
static std::vector<float> g_cmvnScale;
static std::unique_ptr<knf::FbankComputer> g_fbankComputer;
static knf::FeatureWindowFunction g_windowFunction;
static knf::FbankOptions g_fbankOpts;
static bool g_knfInitialized = false;
static int g_knfNMels = DEFAULT_N_MELS;


static bool LoadCMVN(const std::string& cmvnFile)
{
    if (!g_cmvnShift.empty()) {
        return true;
    }
    
    std::ifstream txtFile(cmvnFile);
    if (txtFile.is_open()) {
        std::string line;
        while (std::getline(txtFile, line)) {
            if (line.find("<AddShift>") != std::string::npos || line.find("AddShift") != std::string::npos) {
                std::getline(txtFile, line);
                size_t start = line.find("[");
                size_t end = line.find("]");
                if (start != std::string::npos && end != std::string::npos && end > start) {
                    std::istringstream iss(line.substr(start + 1, end - start - 1));
                    float v;
                    while (iss >> v) {
                        g_cmvnShift.push_back(v);
                    }
                }
            } else if (line.find("<Rescale>") != std::string::npos || line.find("Rescale") != std::string::npos) {
                std::getline(txtFile, line);
                size_t start = line.find("[");
                size_t end = line.find("]");
                if (start != std::string::npos && end != std::string::npos && end > start) {
                    std::istringstream iss(line.substr(start + 1, end - start - 1));
                    float v;
                    while (iss >> v) {
                        g_cmvnScale.push_back(v);
                    }
                }
            }
        }
        txtFile.close();
    } else {
        std::ifstream binFile("../data/am.mvn.bin", std::ios::binary);
        if (!binFile.is_open()) {
            LOG(ERROR) << "CMVN file not found: " << cmvnFile;
            return false;
        }
        int cmvnDim;
        binFile.read(reinterpret_cast<char*>(&cmvnDim), sizeof(int));
        g_cmvnShift.resize(cmvnDim);
        g_cmvnScale.resize(cmvnDim);
        binFile.read(reinterpret_cast<char*>(g_cmvnShift.data()), cmvnDim * sizeof(float));
        binFile.read(reinterpret_cast<char*>(g_cmvnScale.data()), cmvnDim * sizeof(float));
        binFile.close();
    }
    
    return true;
}

static void InitKNF(int nMels, int frameLengthMs, int frameShiftMs, 
                    float preemphCoeff, float dither, const std::string& windowType)
{
    if (g_knfInitialized) {
        return;
    }
    
    g_fbankOpts.frame_opts.samp_freq = SAMPLE_RATE;
    g_fbankOpts.frame_opts.frame_length_ms = frameLengthMs;
    g_fbankOpts.frame_opts.frame_shift_ms = frameShiftMs;
    g_fbankOpts.frame_opts.dither = dither;
    g_fbankOpts.frame_opts.preemph_coeff = preemphCoeff;
    g_fbankOpts.frame_opts.window_type = windowType;
    g_fbankOpts.frame_opts.remove_dc_offset = true;
    g_fbankOpts.frame_opts.snip_edges = true;
    g_fbankOpts.mel_opts.num_bins = nMels;
    g_fbankOpts.mel_opts.low_freq = LOW_FREQ;
    g_fbankOpts.mel_opts.high_freq = 0.0f;
    g_fbankOpts.use_log_fbank = true;
    g_fbankOpts.use_power = true;
    
    g_fbankComputer = std::make_unique<knf::FbankComputer>(g_fbankOpts);
    g_windowFunction = knf::FeatureWindowFunction(g_fbankOpts.frame_opts);
    g_knfNMels = nMels;
    g_knfInitialized = true;
}

static bool LoadWavToBuffer(const std::string& wavPath, float* buffer, size_t bufferSize, size_t& numSamples)
{
    std::ifstream file(wavPath, std::ios::binary);
    if (!file.is_open()) {
        LOG(ERROR) << "WAV not found: " << wavPath;
        return false;
    }

    char riff[WAV_CHUNK_ID_SIZE], wave[WAV_CHUNK_ID_SIZE];
    uint32_t fileSize;
    file.read(riff, WAV_CHUNK_ID_SIZE);
    file.read(reinterpret_cast<char*>(&fileSize), WAV_CHUNK_ID_SIZE);
    file.read(wave, WAV_CHUNK_ID_SIZE);

    if (std::memcmp(riff, "RIFF", WAV_CHUNK_ID_SIZE) != 0 || std::memcmp(wave, "WAVE", WAV_CHUNK_ID_SIZE) != 0) {
        LOG(ERROR) << "Invalid WAV";
        return false;
    }

    uint16_t audioFormat = 0, numChannels = 0, bitsPerSample = 0;
    uint32_t sampleRate = 0;

    while (!file.eof()) {
        char chunkId[WAV_CHUNK_ID_SIZE];
        uint32_t chunkSize;
        if (!file.read(chunkId, WAV_CHUNK_ID_SIZE)) {
            break;
        }
        file.read(reinterpret_cast<char*>(&chunkSize), WAV_CHUNK_ID_SIZE);

        if (std::memcmp(chunkId, "fmt ", WAV_CHUNK_ID_SIZE) == 0) {
            file.read(reinterpret_cast<char*>(&audioFormat), 2);
            file.read(reinterpret_cast<char*>(&numChannels), 2);
            file.read(reinterpret_cast<char*>(&sampleRate), WAV_CHUNK_ID_SIZE);
            uint32_t byteRate;
            uint16_t blockAlign;
            file.read(reinterpret_cast<char*>(&byteRate), WAV_CHUNK_ID_SIZE);
            file.read(reinterpret_cast<char*>(&blockAlign), 2);
            file.read(reinterpret_cast<char*>(&bitsPerSample), 2);
            if (chunkSize > 16) {
                file.seekg(chunkSize - 16, std::ios::cur);
            }
        } else if (std::memcmp(chunkId, "data", WAV_CHUNK_ID_SIZE) == 0) {
            if (audioFormat != 1 || sampleRate != SAMPLE_RATE || bitsPerSample != BITS_PER_SAMPLE) {
                LOG(ERROR) << "Unsupported WAV";
                return false;
            }

            uint32_t numPcm = chunkSize / sizeof(int16_t);
            numSamples = numPcm / numChannels;
            
            if (numSamples > bufferSize) {
                LOG(ERROR) << "WAV too large: " << numSamples << " > buffer " << bufferSize;
                return false;
            }
            
            std::vector<int16_t> pcmBuffer(PCM_BUFFER_SIZE);
            size_t samplesRead = 0;
            
            while (samplesRead < numSamples) {
                size_t toRead = std::min((size_t)PCM_BUFFER_SIZE, numPcm - samplesRead * numChannels);
                file.read(reinterpret_cast<char*>(pcmBuffer.data()), toRead * sizeof(int16_t));
                
                for (size_t i = 0; i < toRead / numChannels && samplesRead < numSamples; i++) {
                    buffer[samplesRead++] = static_cast<float>(pcmBuffer[i * numChannels]);
                }
            }
            
            file.close();
            return true;
        } else {
            file.seekg(chunkSize, std::ios::cur);
        }
    }

    LOG(ERROR) << "No data chunk";
    return false;
}

static void ExtractWindowDirect(const float* waveform, size_t numSamples, int frameIdx,
    const knf::FrameExtractionOptions& opts, const knf::FeatureWindowFunction& g_windowFunction,
    std::vector<float>* signalFrame)
{
    int frameLength = opts.WindowSize();
    int paddedSize = opts.PaddedWindowSize();
    
    if (signalFrame->size() != paddedSize) {
        signalFrame->resize(paddedSize);
    }
    
    int64_t startSample = knf::FirstSampleOfFrame(frameIdx, opts);
    int waveStart = static_cast<int>(startSample);
    int waveEnd = waveStart + frameLength;
    
    if (waveStart >= 0 && waveEnd <= static_cast<int>(numSamples)) {
        std::memcpy(signalFrame->data(), waveform + waveStart, frameLength * sizeof(float));
    } else {
        for (int s = 0; s < frameLength; ++s) {
            int sInWave = s + waveStart;
            while (sInWave < 0 || sInWave >= static_cast<int>(numSamples)) {
                if (sInWave < 0)
                    sInWave = -sInWave - 1;
                else
                    sInWave = 2 * static_cast<int>(numSamples) - 1 - sInWave;
            }
            (*signalFrame)[s] = waveform[sInWave];
        }
    }
    
    for (int i = frameLength; i < paddedSize; i++) {
        (*signalFrame)[i] = 0.0f;
    }
    
    knf::ProcessWindow(opts, g_windowFunction, signalFrame->data(), nullptr);
}

static bool WavFrontendPipelineKNF(const WavFrontendInput& input, WavFrontendOutput& output, const LfrConfig& lfr)
{
    int numFbankFrames = knf::NumFrames(static_cast<int64_t>(input.numSamples), g_fbankOpts.frame_opts, true);
    if (numFbankFrames <= 0) {
        LOG(ERROR) << "No fbank frames computed";
        return false;
    }
    
    int leftPad = (lfr.m - 1) / 2;
    int TLfr = (numFbankFrames + lfr.n - 1) / lfr.n;
    output.numFrames = std::min((size_t)TLfr, output.maxFrames);
    
    int ringSize = lfr.m + 2;
    std::vector<float> fbankRingBuffer(ringSize * g_knfNMels);
    std::vector<float> signalFrame(g_fbankOpts.frame_opts.PaddedWindowSize());
    std::vector<float> feature(g_knfNMels);
    
    size_t lfrOutputCount = 0;
    
    for (int frameIdx = 0; frameIdx < numFbankFrames; frameIdx++) {
        ExtractWindowDirect(input.waveform, input.numSamples, frameIdx, g_fbankOpts.frame_opts, g_windowFunction, &signalFrame);
        
        g_fbankComputer->Compute(0.0f, 1.0f, &signalFrame, feature.data());
        
        int ringIdx = frameIdx % ringSize;
        std::memcpy(fbankRingBuffer.data() + ringIdx * g_knfNMels, feature.data(), 
                    g_knfNMels * sizeof(float));
        
        if ((frameIdx + 1) % lfr.n == 0 && lfrOutputCount < output.numFrames) {
            int lfrI = frameIdx / lfr.n;
            float* dest = output.data + lfrI * output.featDim;
            
            for (int j = 0; j < lfr.m; j++) {
                int inputIdx = lfrI * lfr.n + j;
                int cacheIdx;
                if (inputIdx < leftPad) {
                    cacheIdx = 0;
                } else {
                    cacheIdx = inputIdx - leftPad;
                    cacheIdx = std::min(cacheIdx, numFbankFrames - 1);
                }
                int ringCacheIdx = cacheIdx % ringSize;
                std::memcpy(dest + j * g_knfNMels, fbankRingBuffer.data() + ringCacheIdx * g_knfNMels, 
                    g_knfNMels * sizeof(float));
            }
            
            if (!g_cmvnShift.empty() && !g_cmvnScale.empty()) {
                int applyDim = std::min((int)output.featDim, (int)g_cmvnShift.size());
                for (int d = 0; d < applyDim; d++) {
                    dest[d] = (dest[d] + g_cmvnShift[d]) * g_cmvnScale[d];
                }
            }
            
            lfrOutputCount++;
        }
    }
    
    if (lfrOutputCount < output.numFrames && numFbankFrames % lfr.n != 0) {
        int lfrI = numFbankFrames / lfr.n;
        float* dest = output.data + lfrI * output.featDim;
        
        for (int j = 0; j < lfr.m; j++) {
            int inputIdx = lfrI * lfr.n + j;
            int cacheIdx;
            if (inputIdx < leftPad) {
                cacheIdx = 0;
            } else {
                cacheIdx = inputIdx - leftPad;
                cacheIdx = std::min(cacheIdx, numFbankFrames - 1);
            }
            int ringCacheIdx = cacheIdx % ringSize;
            std::memcpy(dest + j * g_knfNMels, fbankRingBuffer.data() + ringCacheIdx * g_knfNMels, 
                g_knfNMels * sizeof(float));
        }
        
        if (!g_cmvnShift.empty() && !g_cmvnScale.empty()) {
            int applyDim = std::min((int)output.featDim, (int)g_cmvnShift.size());
            for (int d = 0; d < applyDim; d++) {
                dest[d] = (dest[d] + g_cmvnShift[d]) * g_cmvnScale[d];
            }
        }
    }
    
    return true;
}

bool XiaoYunPreprocess(std::vector<std::string>& fileList,  std::vector<TensorBuf>& inBufs,  std::vector<TensorDesc>& inDescs)
{
    if (fileList.empty() || inBufs.empty() || inDescs.empty()) {
        LOG(ERROR) << "invalid input";
        return false;
    }

    TensorBuf& inTensor = inBufs[0];
    TensorBuf tmpBuf;
    if (inBufs.size() == 4) {
        tmpBuf = inBufs[inBufs.size() - 2];
    } else {
        tmpBuf = inBufs[inBufs.size() - 1];
    }

    std::string cmvnPath = "../data/am.mvn.dim80_l2r2";
    std::string wavPath = fileList[0];
    
    int lfrM = DEFAULT_LFR_M;
    int lfrN = DEFAULT_LFR_N;
    int nMels = DEFAULT_N_MELS;
    float dither = 0.0f;
    std::string windowType = "hamming";
    
    InitKNF(nMels, DEFAULT_FRAME_LENGTH_MS, DEFAULT_FRAME_SHIFT_MS, DEFAULT_PREEMPH_COEFF, dither, windowType);
    LoadCMVN(cmvnPath);
    
    size_t modelFrames = inDescs[0].dims[1];
    size_t modelDim = inDescs[0].dims[2];
    std::memset(inTensor.GetRawPtr(), 0, inTensor.size);

    float* samplesPtr = static_cast<float*>(tmpBuf.GetRawPtr());
    size_t tmpBufCapacity = tmpBuf.size / sizeof(float);
    size_t numSamples = 0;
    if (!LoadWavToBuffer(wavPath, samplesPtr, tmpBufCapacity, numSamples)) {
        LOG(ERROR) << "WAV load failed";
        return false;
    }

    float* out = static_cast<float*>(inTensor.GetRawPtr());
    size_t numFrames = 0;
    
    WavFrontendInput input{samplesPtr, numSamples};
    WavFrontendOutput output{out, modelFrames, modelDim, 0};
    LfrConfig lfr{lfrM, lfrN};
    
    if (!WavFrontendPipelineKNF(input, output, lfr)) {
        LOG(ERROR) << "feature extraction failed";
        return false;
    }
    
    numFrames = output.numFrames;

    if (numFrames < modelFrames) {
        float* lastFrame = out + (numFrames - 1) * modelDim;
        for (size_t padFrame = numFrames; padFrame < modelFrames; padFrame++) {
            float* dest = out + padFrame * modelDim;
            std::memcpy(dest, lastFrame, modelDim * sizeof(float));
        }
    }
    
    return true;
}

}
}