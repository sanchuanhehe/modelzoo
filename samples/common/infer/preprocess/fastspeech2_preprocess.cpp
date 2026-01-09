/*
 * Copyright (c) ModelZoo. 2025-2025. All rights reserved.
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

#include "fastspeech2_preprocess.h"
#include "dev_interface_adapter.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <codecvt>
#include <fstream>
#include <locale>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace Infer {
namespace FastSpeech2NS {
    // 常量定义
    constexpr char PAD_SYMBOL[] = "sp";
    constexpr char PUNCTUATION[] = "\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    constexpr char PAD_CHAR = '_';
    constexpr char SPECIAL_CHAR = '-';
    constexpr char PUNCTUATION_CHARS[] = "!'(),.:;? ";
    constexpr char LETTERS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    constexpr char SILENCES[][5] = { "@sp", "@spn", "@sil" };
    constexpr int SILENCES_COUNT = 3;
    constexpr int BYTE_BIT_NUM = 8;
    constexpr int UINT16_BIT_NUM = 16;

    // 音素符号集合
    static const std::vector<std::string> initials = {
        "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n",
        "p", "q", "r", "s", "sh", "t", "w", "x", "y", "z", "zh"
    };

    static const std::vector<std::string> finals = {
        "a1", "a2", "a3", "a4", "a5", "ai1", "ai2", "ai3", "ai4", "ai5",
        "an1", "an2", "an3", "an4", "an5", "ang1", "ang2", "ang3", "ang4", "ang5",
        "ao1", "ao2", "ao3", "ao4", "ao5", "e1", "e2", "e3", "e4", "e5",
        "ei1", "ei2", "ei3", "ei4", "ei5", "en1", "en2", "en3", "en4", "en5",
        "eng1", "eng2", "eng3", "eng4", "eng5", "er1", "er2", "er3", "er4", "er5",
        "i1", "i2", "i3", "i4", "i5", "ia1", "ia2", "ia3", "ia4", "ia5",
        "ian1", "ian2", "ian3", "ian4", "ian5", "iang1", "iang2", "iang3", "iang4", "iang5",
        "iao1", "iao2", "iao3", "iao4", "iao5", "ie1", "ie2", "ie3", "ie4", "ie5",
        "ii1", "ii2", "ii3", "ii4", "ii5", "iii1", "iii2", "iii3", "iii4", "iii5",
        "in1", "in2", "in3", "in4", "in5", "ing1", "ing2", "ing3", "ing4", "ing5",
        "iong1", "iong2", "iong3", "iong4", "iong5", "iou1", "iou2", "iou3", "iou4", "iou5",
        "o1", "o2", "o3", "o4", "o5", "ong1", "ong2", "ong3", "ong4", "ong5",
        "ou1", "ou2", "ou3", "ou4", "ou5", "u1", "u2", "u3", "u4", "u5",
        "ua1", "ua2", "ua3", "ua4", "ua5", "uai1", "uai2", "uai3", "uai4", "uai5",
        "uan1", "uan2", "uan3", "uan4", "uan5", "uang1", "uang2", "uang3", "uang4", "uang5",
        "uei1", "uei2", "uei3", "uei4", "uei5", "uen1", "uen2", "uen3", "uen4", "uen5",
        "uo1", "uo2", "uo3", "uo4", "uo5", "v1", "v2", "v3", "v4", "v5",
        "van1", "van2", "van3", "van4", "van5", "ve1", "ve2", "ve3", "ve4", "ve5",
        "vn1", "vn2", "vn3", "vn4", "vn5"
    };

    static const std::vector<std::string> cmudictValidSymbols = {
        "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2",
        "AO", "AO0", "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
        "B", "CH", "D", "DH", "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2",
        "EY", "EY0", "EY1", "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY",
        "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1", "OW2",
        "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH", "UH0", "UH1",
        "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"
    };

    // 符号映射表
    static std::unordered_map<std::string, int> symbolToId;
    static std::unordered_map<int, std::string> idToSymbol;

    // 初始化符号映射表
    static void InitSymbolMaps()
    {
        if (!symbolToId.empty())
            return;

        std::vector<std::string> symbols;

        // 1. Pad字符 (_)
        symbols.push_back(std::string(1, PAD_CHAR));
        // 2. Special字符 (-)
        symbols.push_back(std::string(1, SPECIAL_CHAR));
        // 3. 标点符号（跳过\0结束符）
        for (char c : PUNCTUATION_CHARS) {
            if (c == '\0')
                break;
            symbols.push_back(std::string(1, c));
        }
        // 4. 大小写字母（跳过\0结束符）
        for (char c : LETTERS) {
            if (c == '\0')
                break;
            symbols.push_back(std::string(1, c));
        }

        // ARPAbet符号
        std::vector<std::string> arpabet;
        for (const auto& s : cmudictValidSymbols) {
            arpabet.push_back("@" + s);
        }
        symbols.insert(symbols.end(), arpabet.begin(), arpabet.end());

        // 拼音符号
        std::vector<std::string> pinyin;
        for (const auto& s : initials) {
            pinyin.push_back("@" + s);
        }
        for (const auto& s : finals) {
            pinyin.push_back("@" + s);
        }
        pinyin.push_back("@rr");
        symbols.insert(symbols.end(), pinyin.begin(), pinyin.end());

        // 静音符号
        for (int i = 0; i < SILENCES_COUNT; ++i) {
            symbols.push_back(SILENCES[i]);
        }

        // 构建映射表
        for (size_t i = 0; i < symbols.size(); ++i) {
            symbolToId[symbols[i]] = i;
            idToSymbol[i] = symbols[i];
        }
    }

    // 全局拼音映射表
    static std::unordered_map<std::wstring, std::string> CHINESE_TO_PINYIN;

    // 加载拼音字典文件
    static bool LoadPinyinDict(const std::string& dictPath)
    {
        std::ifstream file(dictPath);
        if (!file.is_open()) {
            LOG(ERROR) << "Failed to open pinyin dict file: " << dictPath;
            return false;
        }

        // 转换UTF-8到wstring的工具
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        std::string line;

        while (std::getline(file, line)) {
            // 分割行：汉字\t拼音
            size_t tabPos = line.find('\t');
            if (tabPos == std::string::npos) {
                continue;
            }

            // 提取汉字（UTF-8转wstring）
            std::string charStr = line.substr(0, tabPos);
            std::wstring wchar = conv.from_bytes(charStr);

            // 提取拼音
            std::string pinyin = line.substr(tabPos + 1);

            // 加入映射表
            CHINESE_TO_PINYIN[wchar] = pinyin;
        }

        file.close();
        LOG(INFO) << "Loaded pinyin dict, total chars: " << CHINESE_TO_PINYIN.size();
        return true;
    }

    // 读取词典文件
    static std::unordered_map<std::string, std::vector<std::string>> ReadLexicon(const std::string& lexPath)
    {
        std::unordered_map<std::string, std::vector<std::string>> lexicon;
        std::ifstream file(lexPath);
        if (!file.is_open()) {
            LOG(ERROR) << "Failed to open lexicon file: " << lexPath;
            return lexicon;
        }

        std::string line;
        while (std::getline(file, line)) {
            // 移除行尾换行符
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            // 分割单词和音素
            std::vector<std::string> parts;
            size_t pos = 0;
            while (pos < line.size()) {
                size_t nextPos = line.find_first_of(" \t", pos);
                if (nextPos == std::string::npos) {
                    parts.push_back(line.substr(pos));
                    break;
                }
                if (nextPos > pos) {
                    parts.push_back(line.substr(pos, nextPos - pos));
                }
                pos = nextPos + 1;
            }

            if (parts.size() >= 2) {
                std::string word = parts[0];
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                std::vector<std::string> phones(parts.begin() + 1, parts.end());
                // 仅当单词未存在时才添加（保证首次出现的音素被保留）
                if (lexicon.find(word) == lexicon.end()) {
                    lexicon[word] = phones;
                }
            }
        }

        file.close();
        LOG(INFO) << "lexicon size is " << lexicon.size();
        return lexicon;
    }

    // 填充音素序列至目标长度
    static std::vector<std::string> PadPhonesWithSp(const std::vector<std::string>& phones, int targetLen)
    {
        std::vector<std::string> paddedPhones;

        if (phones.size() >= static_cast<size_t>(targetLen)) {
            paddedPhones = std::vector<std::string>(phones.begin(), phones.begin() + targetLen);
        } else {
            paddedPhones = phones;
            int padNum = targetLen - phones.size();
            for (int i = 0; i < padNum; ++i) {
                paddedPhones.push_back(PAD_SYMBOL);
            }
        }

        return paddedPhones;
    }

    // 预处理英文文本
    static std::string PreprocessEnglish(const std::string& text, const std::unordered_map<std::string, std::vector<std::string>>& lexicon, int targetPhoneLength)
    {
        // 1. 仅移除末尾标点
        std::string processedText = text;
        size_t lastNonPunct = processedText.find_last_not_of(PUNCTUATION);
        if (lastNonPunct != std::string::npos) {
            processedText = processedText.substr(0, lastNonPunct + 1);
        }

        // 2. 按正则逻辑分词
        std::regex splitRegex(R"([,;.\-\?\!\s+])");
        std::sregex_token_iterator it(processedText.begin(), processedText.end(), splitRegex, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> words;
        for (; it != end; ++it) {
            words.push_back(*it);
        }
        LOG(INFO) << "words size is " << words.size();

        // 3. 转换为音素
        std::vector<std::string> phones;
        for (const auto& w : words) {
            if (w.empty()) {
                continue;
            }
            std::string lowerWord = w;
            std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);

            auto lexIt = lexicon.find(lowerWord);
            if (lexIt != lexicon.end()) {
                phones.insert(phones.end(), lexIt->second.begin(), lexIt->second.end());
            } else {
                phones.push_back(PAD_SYMBOL);
            }
        }

        // 4. 填充至目标长度
        std::vector<std::string> paddedPhones = PadPhonesWithSp(phones, targetPhoneLength);

        // 5. 格式化音素字符串
        std::string phoneStr = "{";
        for (size_t i = 0; i < paddedPhones.size(); ++i) {
            if (i > 0) {
                phoneStr += " ";
            }
            phoneStr += paddedPhones[i];
        }
        phoneStr += "}";

        // 6. 替换无效符号
        std::regex invalidRe(R"(\{[^\w\s]+\})");
        phoneStr = std::regex_replace(phoneStr, invalidRe, "{sp}");

        LOG(INFO) << "原始文本：" << text << "，转换为音素：" << phoneStr;
        return phoneStr;
    }

    // 预处理中文文本（完全对齐Python的preprocess_mandarin逻辑）
    static std::string PreprocessMandarin(const std::string& text, const std::unordered_map<std::string, std::vector<std::string>>& lexicon, int targetPhoneLength)
    {
        // 1. 转换UTF-8字符串到宽字符（处理中文）
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        std::wstring wtext = conv.from_bytes(text);

        // 2. 逐字获取拼音（对齐Python: Style.TONE3 + strict=False + neutral_tone_with_five=True）
        std::vector<std::string> pinyins;
        for (const wchar_t& wc : wtext) {
            // 跳过非中文字符（兼容Python的宽松模式）
            if (wc < 0x4E00 || wc > 0x9FFF) {
                continue;
            }

            // 从预加载的拼音字典中查找拼音
            std::string py = "sp"; // 默认sp（对齐Python的else逻辑）
            auto pyIt = CHINESE_TO_PINYIN.find(std::wstring(1, wc));
            if (pyIt != CHINESE_TO_PINYIN.end()) {
                py = pyIt->second;
            }
            pinyins.push_back(py);
        }

        // 3. 拼音转音素（对齐Python逻辑：查lexicon，不存在则填sp）
        std::vector<std::string> phones;
        for (const std::string& p : pinyins) {
            auto lexIt = lexicon.find(p);
            if (lexIt != lexicon.end()) {
                // 拼音存在于词典，添加对应的音素列表
                phones.insert(phones.end(), lexIt->second.begin(), lexIt->second.end());
            } else {
                // 拼音不存在，添加sp（静音符号）
                phones.push_back(PAD_SYMBOL);
            }
        }

        // 4. 填充音素至目标长度（对齐Python的pad_phones_with_sp）
        std::vector<std::string> paddedPhones = PadPhonesWithSp(phones, targetPhoneLength);

        // 5. 格式化音素字符串（对齐Python: "{音素1 音素2 ...}"）
        std::string phoneStr = "{";
        for (size_t i = 0; i < paddedPhones.size(); ++i) {
            if (i > 0) {
                phoneStr += " ";
            }
            phoneStr += paddedPhones[i];
        }
        phoneStr += "}";

        LOG(INFO) << "原始中文文本：" << text << "，转换为音素：" << phoneStr;
        return phoneStr;
    }

    // 文本转换为序列
    static std::vector<int64_t> TextToSequence(const std::string& text)
    {
        std::vector<int64_t> sequence;
        std::regex curlyRe(R"((.*?)\{(.+?)\}(.*))");
        std::smatch match;
        std::string remainingText = text;

        while (std::regex_match(remainingText, match, curlyRe)) {
            std::string prefix = match[1];
            std::string arpabet = match[2];
            remainingText = match[3];

            // 处理ARPAbet部分
            std::vector<std::string> arpabetSymbols;
            size_t pos = 0;
            while (pos < arpabet.size()) {
                size_t nextPos = arpabet.find(' ', pos);
                if (nextPos == std::string::npos) {
                    arpabetSymbols.push_back(arpabet.substr(pos));
                    break;
                }
                if (nextPos > pos) {
                    arpabetSymbols.push_back(arpabet.substr(pos, nextPos - pos));
                }
                pos = nextPos + 1;
            }

            for (const auto& symbol : arpabetSymbols) {
                std::string key = "@" + symbol;
                auto it = symbolToId.find(key);
                if (it != symbolToId.end()) {
                    sequence.push_back(it->second);
                }
            }
        }

        // ========== 新增打印信息 ==========
        LOG(INFO) << "[TextToSequence] 原始音素: " << text;
        LOG(INFO) << "[TextToSequence] 转换后的整数序列 (长度: " << sequence.size() << "): ";
        std::string seqStr;
        for (size_t i = 0; i < sequence.size(); ++i) {
            if (i > 0) {
                seqStr += ", ";
            }
            seqStr += std::to_string(sequence[i]);
        }
        LOG(INFO) << "[TextToSequence] " << seqStr;
        // =================================

        return sequence;
    }

    // 解析单行文本
    bool ParseLine(const std::string& line, std::string& textId, std::string& text)
    {
        std::vector<std::string> parts;
        size_t pos = 0;
        size_t delimiterCount = 0;

        while (pos < line.size() && delimiterCount < 4) {
            size_t nextPos = line.find('|', pos);
            if (nextPos == std::string::npos) {
                parts.push_back(line.substr(pos));
                break;
            }
            parts.push_back(line.substr(pos, nextPos - pos));
            pos = nextPos + 1;
            delimiterCount++;
        }

        if (parts.size() < 4) {
            LOG(WARNING) << "Invalid line format: " << line;
            return false;
        }

        textId = parts[0];
        text = parts[3];
        return true;
    }

    /**
     * @brief 将std::vector<int64_t>转换为std::vector<uint16_t>
     * @param src 源容器（int64_t类型）
     * @return 转换后的uint16_t类型容器（超出范围的值替换为边界值，打印ERROR日志）
     */
    std::vector<uint16_t> convertInt64VectorToUint16(const std::vector<int64_t>& src)
    {
        // 初始化目标容器，提前预留和源容器相同的空间，避免频繁扩容
        std::vector<uint16_t> dst;
        dst.reserve(src.size());

        for (size_t i = 0; i < src.size(); ++i) {
            // 获取当前要转换的int64_t值
            int64_t current_num = src[i];
            uint16_t converted_num = 0;

            // 1. 处理负数（替换为uint16_t最小值0）
            if (current_num < 0) {
                converted_num = 0;
                // 使用项目自定义的LOG(ERROR)打印提示，包含索引和具体值，方便定位
                LOG(ERROR) << "convertInt64VectorToUint16: index " << i
                           << " value " << current_num << " is negative, replaced with 0 (uint16_t min).";
            }
            // 2. 处理超过uint16_t最大值的情况（替换为65535）
            else if (current_num > UINT16_MAX) {
                converted_num = UINT16_MAX;
                LOG(ERROR) << "convertInt64VectorToUint16: index " << i
                           << " value " << current_num << " exceeds uint16_t max (" << UINT16_MAX
                           << "), replaced with " << UINT16_MAX << ".";
            }
            // 3. 正常范围，直接转换
            else {
                converted_num = static_cast<uint16_t>(current_num);
            }

            // 将转换后的值加入目标容器
            dst.push_back(converted_num);
        }

        return dst;
    }

    /**
     * @brief 将UINT16类型的音素序列拷贝到模型输入缓冲区
     * @param sequenceUInt16: 预处理后的UINT16类型音素序列（一维）
     * @param desc: 模型输入Tensor的描述信息（维度、类型、大小等）
     * @param inBuf: 模型输入缓冲区（需要写入数据的目标缓冲区）
     * @return Result: SUCCESS/FAILED，拷贝成功/失败
     */
    static Result ReadUInt16SeqToBuf(const std::vector<uint16_t>& sequenceUInt16,
        TensorDesc& desc,
        TensorBuf& inBuf)
    {
        // 1. 基础参数校验
        LOG(INFO) << "ReadUInt16SeqToBuf: desc.dimCount = " << desc.dimCount
                  << ", desc.typeSize = " << desc.typeSize
                  << ", desc.defaultSize = " << desc.defaultSize;

        // 校验输入序列非空
        if (sequenceUInt16.empty()) {
            LOG(ERROR) << "Input UINT16 sequence is empty!";
            return FAILED;
        }

        // 校验数据类型匹配（必须是UINT16）
        if (desc.typeSize != UINT16_BIT_NUM) {
            LOG(ERROR) << "Unsupported tensor type size! Expect UINT16(" << UINT16_BIT_NUM
                       << "), actual = " << desc.typeSize;
            return FAILED;
        }

        // 校验缓冲区大小足够
        size_t seqDataSize = sequenceUInt16.size() * sizeof(uint16_t);
        if (seqDataSize > desc.defaultSize) {
            LOG(ERROR) << "Sequence data size exceeds buffer size! Seq size = " << seqDataSize
                       << ", buffer size = " << desc.defaultSize;
            return FAILED;
        }

        // 2. 获取源数据指针和目标缓冲区指针
        const uint16_t* srcData = sequenceUInt16.data();
        void* destRawPtr = inBuf.GetRawPtr();
        if (srcData == nullptr || destRawPtr == nullptr) {
            LOG(ERROR) << "Null pointer detected! srcData = " << (void*)srcData
                       << ", destRawPtr = " << destRawPtr;
            return FAILED;
        }
        uint16_t* destData = static_cast<uint16_t*>(destRawPtr);

        // 3. 分两种场景处理拷贝（兼容dlite/dpico核）
        // 场景1: dlite核（无字节对齐，直接拷贝）
        if (desc.defaultStride == 0) {
            LOG(INFO) << "Copy UINT16 sequence to dlite core buffer, size = " << seqDataSize;
            memcpy(destData, srcData, seqDataSize);
            // 若缓冲区有剩余空间，填充0（可选，根据模型要求）
            if (desc.defaultSize > seqDataSize) {
                size_t padSize = desc.defaultSize - seqDataSize;
                memset(destData + sequenceUInt16.size(), 0, padSize);
                LOG(INFO) << "Pad remaining buffer with 0, pad size = " << padSize;
            }
            return SUCCESS;
        }

        // 场景2: dpico核（需要处理字节对齐/步长）
        LOG(INFO) << "Copy UINT16 sequence to dpico core buffer (with stride)";

        // 计算循环次数（一维序列：loopTimes=1，多维可扩展）
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        // 关键参数计算（适配UINT16）
        int64_t width = desc.dims[desc.dimCount - 1]; // 最后一维为序列长度
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM; // UINT16: 16/8=2字节
        size_t strideElemNum = inBuf.stride / dataSize; // 步长对应的元素数

        // 预计算源数据总元素数（避免循环内重复计算）
        size_t srcTotalElem = sequenceUInt16.size();

        // 逐段拷贝（处理步长对齐）
        for (int64_t loop = 0; loop < loopTimes; loop++) {
            // 目标缓冲区当前段指针
            uint16_t* destPtr = destData + loop * strideElemNum;
            // 源数据当前段起始索引（整数）
            int64_t srcStartIdx = loop * width;
            // 源数据当前段指针（通过索引计算，避免指针运算错误）
            const uint16_t* srcPtr = srcData + srcStartIdx;

            // 边界校验（修复核心：全部用整数比较，避免指针vs整数）
            if (destPtr == nullptr || srcPtr == nullptr) {
                LOG(ERROR) << "Null pointer in loop " << loop;
                return FAILED;
            }
            // 校验：当前段起始索引 + 宽度 不超过源数据总元素数
            if (srcStartIdx + width > static_cast<int64_t>(srcTotalElem)) {
                LOG(ERROR) << "Source data out of range in loop " << loop
                           << "! srcStartIdx = " << srcStartIdx
                           << ", width = " << width
                           << ", total elem = " << srcTotalElem;
                return FAILED;
            }

            // 拷贝当前段数据
            memcpy(destPtr, srcPtr, width * dataSize);
            LOG(INFO) << "Loop " << loop << " copy " << width * dataSize << " bytes";
        }

        return SUCCESS;
    }

    // 主预处理函数
    bool FastSpeech2Preprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
    {
        LOG(INFO) << "====== 前处理启动 ======";
        // 用局部静态变量确保InitSymbolMaps()仅执行一次
        static bool isSymbolMapsInited = false;
        if (!isSymbolMapsInited) {
            InitSymbolMaps();
            isSymbolMapsInited = true;
            LOG(INFO) << "Symbol maps initialized once only!";
        }

        // 读取配置文件
        auto cfg = ReadCfgFile("../data/cfg.txt");
        std::string lexiconEnPath = cfg["lexicon_en_path"];
        std::string lexiconZhPath = cfg["lexicon_zh_path"];
        std::string chinesePinyinDictPath = cfg["chinese_pinyin_dict"];
        std::string language = cfg["language"];
        std::string saveBin = cfg["save_preprocess_bin"];
        std::string saveTxt = cfg["save_preprocess_txt"];
        int targetPhoneLength = std::stoi(cfg["target_phone_length"]);
        mode_t oldUmask = umask(0);

        // 创建保存目录
        if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
            LOG(ERROR) << "Create preprocess bin dir failed: " << saveBin;
            return false;
        }
        if (!saveTxt.empty() && !CreateDirectoryRecursive(saveTxt)) {
            LOG(ERROR) << "Create preprocess txt dir failed: " << saveTxt;
            return false;
        }

        // 加载英文/拼音到音素的映射词典（静态变量保证仅加载一次）
        static std::unordered_map<std::string, std::vector<std::string>> lexicon;
        if (lexicon.empty()) {
            std::string lexiconPath;
            if (language == "en") {
                lexiconPath = lexiconEnPath;
            } else if (language == "zh") {
                lexiconPath = lexiconZhPath;
            } else {
                LOG(ERROR) << "Unsupported language: " << language;
                return false;
            }

            lexicon = ReadLexicon(lexiconPath);
            if (lexicon.empty()) {
                LOG(ERROR) << "Failed to load lexicon from: " << lexiconPath;
                return false;
            }
            LOG(INFO) << "Lexicon loaded once from: " << lexiconPath;
        }

        // 加载中文到拼音的映射词典（静态变量保证仅加载一次）
        static bool isPinyinDictLoaded = false;
        if (!isPinyinDictLoaded) {
            if (!LoadPinyinDict(chinesePinyinDictPath)) {
                LOG(ERROR) << "Failed to load chinese_pinyin_dict from: " << chinesePinyinDictPath;
                return false;
            }
            isPinyinDictLoaded = true;
        }

        for (size_t i = 0; i < fileList.size(); ++i) {
            std::string line = fileList[i];

            std::string textId, text;
            if (!ParseLine(line, textId, text)) {
                LOG(ERROR) << "Skip invalid line: " << line;
                continue;
            }

            // 预处理文本
            std::string phones;
            if (language == "en") {
                phones = PreprocessEnglish(text, lexicon, targetPhoneLength);
            } else if (language == "zh") {
                phones = PreprocessMandarin(text, lexicon, targetPhoneLength);
            } else {
                LOG(ERROR) << "Unsupported language: " << language;
                return false;
            }

            // 转换为序列
            std::vector<int64_t> sequence = TextToSequence(phones);
            if (sequence.empty()) {
                LOG(ERROR) << "Failed to convert text to sequence: " << text;
                return false;
            }
            // 转换为uint16数据类型，方便copy到输入缓冲区
            std::vector<uint16_t> sequenceUInt16 = convertInt64VectorToUint16(sequence);
            if (sequenceUInt16.empty()) {
                LOG(ERROR) << "Failed to convert int64 vector to uint16";
                return false;
            }
            // 拷贝到模型输入缓冲区
            Result ret = ReadUInt16SeqToBuf(sequenceUInt16, inDescs[1], inBufs[1]);
            if (ret != SUCCESS) {
                LOG(ERROR) << "Copy UINT16 sequence to buffer failed!";
                return -1;
            }

            // 保存预处理结果
            if (!saveBin.empty()) {
                std::string binPath = saveBin + "/" + textId + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                if (binFile.is_open()) {
                    binFile.write(reinterpret_cast<const char*>(sequence.data()), sequence.size() * sizeof(int64_t));
                    binFile.close();

                    // 设置文件权限
                    if (chmod(binPath.c_str(), 0777) != 0) {
                        LOG(WARNING) << "Failed to set permissions for bin file: " << binPath;
                    }
                    LOG(INFO) << "Saved preprocess bin: " << binPath;
                } else {
                    LOG(WARNING) << "Failed to open bin file: " << binPath;
                }
            }
        }

        umask(oldUmask);
        LOG(INFO) << "====== 前处理完成 ======";
        return true;
    }
}
}