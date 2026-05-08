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

#include "clip_txt_preprocess.h"
#include <algorithm>
#include <cctype>
#include <codecvt>
#include <cstring>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int CONTEXT_LENGTH = 512;
namespace Infer {
class TextTokenizer {
private:
    // ==================== 词汇表相关 ====================
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> invVocab;

    // ==================== 配置参数 ====================
    bool doLowerCase;
    std::string unkToken;
    int maxInputCharsPerWord;

    // 检查字符是否是空白字符（兼容版，无需uchar_traits）
    bool IsWhitespace(char32_t cp) const
    {
        // 基础空白字符
        if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
            return true;
        }
        // Unicode Zs类别（空白分隔符）- 手动判断常用空白符
        const uint32_t whitespaceCodes[] = {
            0x0020, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
            0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000};
        for (uint32_t code : whitespaceCodes) {
            if (cp == code)
                return true;
        }
        return false;
    }

    // 检查字符是否是控制字符（兼容版）
    bool isonCtrol(char32_t cp) const
    {
        // 排除制表符/换行符
        if (cp == U'\t' || cp == U'\n' || cp == U'\r') {
            return false;
        }
        // 控制字符范围（Cc/Cf类别）- 手动判断
        if ((cp >= 0x0000 && cp <= 0x001F) || (cp >= 0x007F && cp <= 0x009F)) {
            return true;
        }
        return false;
    }

    // 检查字符是否是标点符号（兼容版）
    bool isPunctuation(char32_t cp) const
    {
        // ASCII标点
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }
        // Unicode P类别标点 - 手动判断常用中文/英文标点
        const uint32_t punctuationCodes[] = {
            0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
            0x2008, 0x2009, 0x200A, 0x200B, 0x202F, 0x205F, 0x3000, 0x3001,
            0x3002, 0xFF0C, 0xFF1B, 0xFF1A, 0xFF08, 0xFF09, 0xFF01, 0xFF1F,
            0xFF0E, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x00B7};
        for (uint32_t code : punctuationCodes) {
            if (cp == code)
                return true;
        }
        return false;
    }

    // 检查是否为中文字符（逻辑不变）
    bool IsChineseChar(char32_t cp) const
    {
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK 统一表意文字
            (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK 扩展 A
            (cp >= 0x20000 && cp <= 0x2A6DF) || // CJK 扩展 B
            (cp >= 0x2A700 && cp <= 0x2B73F) || // CJK 扩展 C
            (cp >= 0x2B740 && cp <= 0x2B81F) || // CJK 扩展 D
            (cp >= 0x2B820 && cp <= 0x2CEAF) || // CJK 扩展 E
            (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK 兼容表意文字
            (cp >= 0x2F800 && cp <= 0x2FA1F);   // CJK 兼容表意文字补充
    }

    // UTF-8 转 UTF-32
    std::u32string Utf8ToUtf32(const std::string &utf8Str) const
    {
        std::u32string result;
        const char *ptr = utf8Str.c_str();
        const char *end = ptr + utf8Str.length();

        while (ptr < end) {
            uint8_t b = static_cast<uint8_t>(*ptr);
            char32_t cp = 0;
            int bytes = 0;

            if ((b & 0x80) == 0) {
                cp = b;
                bytes = 1;
            } else if ((b & 0xE0) == 0xC0) {
                cp = (b & 0x1F) << 6;
                bytes = 2;
            } else if ((b & 0xF0) == 0xE0) {
                cp = (b & 0x0F) << 12;
                bytes = 3;
            } else if ((b & 0xF8) == 0xF0) {
                cp = (b & 0x07) << 18;
                bytes = 4;
            } else {
                ptr++;
                continue; // 无效UTF-8
            }

            // 读取后续字节
            for (int i = 1; i < bytes && ptr + i < end; i++) {
                uint8_t nextB = static_cast<uint8_t>(*(ptr + i));
                if ((nextB & 0xC0) != 0x80) {
                    cp = 0;
                    break;
                }
                cp |= (nextB & 0x3F) << (6 * (bytes - i - 1));
            }

            if (cp != 0) {
                result.push_back(cp);
            }
            ptr += bytes;
        }

        return result;
    }

    // UTF-32 转 UTF-8
    std::string Utf32ToUtf8(const std::u32string &utf32_str) const
    {
        std::string result;
        for (char32_t cp : utf32_str) {
            if (cp <= 0x7F) {
                result.push_back(static_cast<char>(cp));
            } else if (cp <= 0x7FF) {
                result.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
                result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp <= 0xFFFF) {
                result.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
                result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp <= 0x10FFFF) {
                result.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
                result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }
        return result;
    }

    // 清理文本
    std::u32string CleanText(const std::u32string &text) const
    {
        std::u32string result;
        for (char32_t cp : text) {
            if (cp == 0 || cp == 0xFFFD || isonCtrol(cp)) {
                continue;
            }
            if (IsWhitespace(cp)) {
                result.push_back(U' ');
            } else {
                result.push_back(cp);
            }
        }
        return result;
    }

    // 在中文字符周围添加空格
    std::u32string TokenizeChineseChars(const std::u32string &text) const
    {
        std::u32string result;
        for (char32_t cp : text) {
            if (IsChineseChar(cp)) {
                result.push_back(U' ');
                result.push_back(cp);
                result.push_back(U' ');
            } else {
                result.push_back(cp);
            }
        }
        return result;
    }

    // 空白字符分词
    std::vector<std::string> WhitespaceTokenize(const std::string &text) const
    {
        std::vector<std::string> result;
        std::string CurrentToken;

        std::u32string u32Text = Utf8ToUtf32(text);
        for (char32_t cp : u32Text) {
            if (IsWhitespace(cp)) {
                if (!CurrentToken.empty()) {
                    result.push_back(CurrentToken);
                    CurrentToken.clear();
                }
            } else {
                CurrentToken += Utf32ToUtf8(std::u32string(1, cp));
            }
        }

        if (!CurrentToken.empty()) {
            result.push_back(CurrentToken);
        }

        return result;
    }

    // 去除重音符号（替换uchar_traits，手动过滤重音）
    std::string StripAccents(const std::string &text) const
    {
        std::u32string u32Text = Utf8ToUtf32(text);
        std::u32string result;

        // 重音符号（Mn类别）的常用编码范围
        for (char32_t cp : u32Text) {
            if (!((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
                (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF) ||
                (cp >= 0xFE20 && cp <= 0xFE2F))) {
                result.push_back(cp);
            }
        }

        return Utf32ToUtf8(result);
    }

    // 根据标点分割
    std::vector<std::string> SplitOnPunctuation(const std::string &text) const
    {
        std::vector<std::string> result;
        std::u32string u32Text = Utf8ToUtf32(text);

        size_t i = 0;
        bool start_new_word = true;

        while (i < u32Text.length()) {
            char32_t cp = u32Text[i];
            if (isPunctuation(cp)) {
                result.push_back(Utf32ToUtf8(std::u32string(1, cp)));
                start_new_word = true;
            } else {
                if (start_new_word) {
                    result.push_back("");
                }
                start_new_word = false;
                result.back() += Utf32ToUtf8(std::u32string(1, cp));
            }
            i++;
        }
        return result;
    }

    // 基础分词
    std::vector<std::string> BasicTokenize(const std::string &text) const
    {
        // 1. 转换为UTF-32并清理
        std::u32string u32Text = Utf8ToUtf32(text);
        u32Text = CleanText(u32Text);

        // 2. 处理中文字符
        u32Text = TokenizeChineseChars(u32Text);
        std::string utf8Text = Utf32ToUtf8(u32Text);

        // 3. 初始空白分词
        std::vector<std::string> origTokens = WhitespaceTokenize(utf8Text);

        std::vector<std::string> splitTokens;
        for (const auto &token : origTokens) {
            std::string processedToken = token;

            // 4. 转小写（仅ASCII）
            if (doLowerCase) {
                std::transform(processedToken.begin(), processedToken.end(),
                    processedToken.begin(),
                    [](unsigned char c) { return std::tolower(c); });
                // 去除重音
                processedToken = StripAccents(processedToken);
            }

            // 5. 按标点分割
            auto puncSplitTokens = SplitOnPunctuation(processedToken);
            splitTokens.insert(splitTokens.end(), puncSplitTokens.begin(),
                puncSplitTokens.end());
        }

        // 6. 重新空白分词
        std::string joined;
        for (size_t i = 0; i < splitTokens.size(); ++i) {
            if (i > 0)
                joined += " ";
            joined += splitTokens[i];
        }

        return WhitespaceTokenize(joined);
    }

    std::vector<std::string> WordPieceTokenize(const std::string &token) const
    {
        std::vector<std::string> outputTokens;

        // 检查token长度
        if (token.length() > static_cast<size_t>(maxInputCharsPerWord)) {
            outputTokens.push_back(unkToken);
            return outputTokens;
        }

        bool isBad = false;
        size_t start = 0;
        std::vector<std::string> subTokens;
        std::string tokenStr = token;

        while (start < tokenStr.length()) {
            size_t end = tokenStr.length();
            std::string curSubstr;
            bool found = false;

            // 最长匹配
            while (start < end) {
                std::string substr = tokenStr.substr(start, end - start);
                if (start > 0) {
                    substr = "##" + substr;
                }

                if (vocab.find(substr) != vocab.end()) {
                    curSubstr = substr;
                    found = true;
                    break;
                }
                end--;
            }

            if (!found) {
                isBad = true;
                break;
            }

            subTokens.push_back(curSubstr);
            start = end;
        }

        if (isBad) {
            outputTokens.push_back(unkToken);
        } else {
            outputTokens.insert(outputTokens.end(), subTokens.begin(), subTokens.end());
        }

        return outputTokens;
    }

    // ==================== 词汇表加载 ====================
    std::unordered_map<std::string, int>
    loadVocab(const std::string &vocabFile) const
    {
        std::unordered_map<std::string, int> vocabMap;
        std::ifstream file(vocabFile);
        if (!file.is_open()) {
            LOG(ERROR) << "VOCAB ERROR";
            return vocabMap;
        }

        std::string line;
        int id = 0;

        while (std::getline(file, line)) {
            // 移除行尾空白/换行符
            size_t endPos = line.find_last_not_of(" \t\n\r");
            if (endPos == std::string::npos)
                continue;
            line = line.substr(0, endPos + 1);

            if (!line.empty()) {
                vocabMap[line] = id++;
            }
        }
        file.close();
        return vocabMap;
    }

    // 清理 tokenization 伪影
    std::string CleanUpTokenization(const std::string &text) const
    {
        std::string result = text;

        std::vector<std::pair<std::string, std::string>> replacements = {
            {" .", "."},     {" ?", "?"},     {" !", "!"},   {" ,", ","},
            {" ' ", "'"},    {" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"},
            {" 've", "'ve"}, {" 're", "'re"}};

        for (const auto &repl : replacements) {
            size_t pos = 0;
            while ((pos = result.find(repl.first, pos)) != std::string::npos) {
                result.replace(pos, repl.first.length(), repl.second);
                pos += repl.second.length();
            }
        }

        return result;
    }

public:
    // 构造函数
    TextTokenizer(const std::string &vocabFile = "../data/vocab.txt",
        bool lowerCase = true, const std::string &unknown_token = "[UNK]", int maxChars = 200)
        : doLowerCase(lowerCase), unkToken(unknown_token), maxInputCharsPerWord(maxChars)
    {
        vocab = loadVocab(vocabFile);

        // 构建反向词汇表
        for (const auto &pair : vocab) {
            invVocab[pair.second] = pair.first;
        }
    }

    std::vector<std::string> tokenize(const std::string &text)
    {
        std::vector<std::string> result;

        // 1. BasicTokenizer 分词
        auto basicTokens = BasicTokenize(text);

        // 2. WordPiece 分词
        for (const auto &token : basicTokens) {
            auto subTokens = WordPieceTokenize(token);
            result.insert(result.end(), subTokens.begin(), subTokens.end());
        }

        return result;
    }

    // 转换 tokens 为 IDs
    std::vector<int>
    ConvertTokensToIds(const std::vector<std::string> &tokens) const
    {
        std::vector<int> ids;
        int unkId = 100;
        auto unkIt = vocab.find(unkToken);
        if (unkIt != vocab.end()) {
            unkId = unkIt->second;
        }

        for (const auto &token : tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(unkId);
            }
        }
        return ids;
    }

    // 转换 IDs 为 tokens
    std::vector<std::string>
    ConvertIdsToTokens(const std::vector<int> &ids) const
    {
        std::vector<std::string> tokens;
        for (int id : ids) {
            auto it = invVocab.find(id);
            if (it != invVocab.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unkToken);
            }
        }
        return tokens;
    }

    // 转换 tokens 为字符串
    std::string ConvertTokensToString(const std::vector<std::string> &tokens,
        bool cleanSpaces = true) const
    {
        std::string text;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0)
                text += " ";
            text += tokens[i];
        }

        // 移除 "##" 标记
        size_t pos = 0;
        while ((pos = text.find("##", pos)) != std::string::npos) {
            text.replace(pos, 2, "");
        }

        // 去除首尾空白
        size_t start = text.find_first_not_of(" \t\n\r");
        size_t end = text.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            text = text.substr(start, end - start + 1);
        }

        if (cleanSpaces) {
            text = CleanUpTokenization(text);
        }

        return text;
    }

    // 获取词汇表大小
    size_t vocabSize() const
    {
        return vocab.size();
    }

    // 检查 token 是否在词汇表中
    bool hasToken(const std::string &token) const
    {
        return vocab.find(token) != vocab.end();
    }

    // 获取 token 的 ID
    int GetTokenId(const std::string &token) const
    {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            return it->second;
        }
        auto unkIt = vocab.find(unkToken);
        if (unkIt != vocab.end()) {
            return unkIt->second;
        }
        return -1;
    }

    // 获取 ID 对应的 token
    std::string GetIdToken(int id) const {
        auto it = invVocab.find(id);
        if (it != invVocab.end()) {
            return it->second;
        }
        return unkToken;
    }
};

// 文本预处理函数：转小写 + 替换中文引号为英文引号
std::string PreprocessTextClip(const std::string &text)
{
    std::string processedText = text;
    for (auto& pattern : {"“", "”"}) {
        size_t pos = 0;
        while ((pos = processedText.find(pattern, pos)) != std::string::npos) {
            processedText.replace(pos, strlen(pattern), "\"");
            pos += 1;
        }
    }
    return processedText;
}

bool ReadSingleToBuf(std::vector<int64_t> allTokens, TensorDesc &desc,
    TensorBuf &inBuf, bool isDpico)
{
    int BYTE_BIT_NUM = 8;

    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
    int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
    // 步骤1：将int64_t的token序列转换为float32
    std::vector<float> floatTokens(allTokens.begin(), allTokens.end());

    // 步骤2：验证维度合法性（desc需匹配[1, context_length]）
    if (desc.dimCount < 2) {
        LOG(ERROR) << "Error: TensorDesc dimCount must be >=2";
        return false;
    }

    // 检查类型是否为float32（typeSize=32比特）
    if (desc.typeSize != 32 && isDpico) {
        LOG(ERROR) << "Error: TensorDesc typeSize must be 32 (float32)";
        return false;
    }

    // 步骤5：将float数据写入inBuf
    if (isDpico) {
        memcpy(static_cast<char *>(inBuf.GetRawPtr()), floatTokens.data(),
            width * dataSize);
    } else {
        memcpy(static_cast<char *>(inBuf.GetRawPtr()), allTokens.data(),
            width * dataSize);
    }
    return true;
}

bool ClipTxtPreprocess(std::vector<std::string> &fileList,
                       std::vector<TensorBuf> &inBufs,
                       std::vector<TensorDesc> &inDescs,
                       bool isDpico)
{
    try {
        LOG(INFO) << "pre txt : " << fileList[0];
        // 创建分词器
        static TextTokenizer tokenizer("../data/vocab.txt", true);
        // 1. 添加[CLS]标记
        std::vector<int64_t> tokens;
        std::vector<std::string> cls_token = {"[CLS]"};
        tokens.push_back(tokenizer.ConvertTokensToIds(cls_token)[0]);

        // 2. 分词并转换为ID，截断到context_length-2（预留CLS和SEP）
        auto text_tokens = tokenizer.tokenize(fileList[0]);
        auto text_ids = tokenizer.ConvertTokensToIds(text_tokens);
        int max_content_len = CONTEXT_LENGTH - 2;
        if (text_ids.size() > max_content_len) {
            text_ids.resize(max_content_len);
        }
        tokens.insert(tokens.end(), text_ids.begin(), text_ids.end());

        // 3. 添加[SEP]标记
        std::vector<std::string> sep_token = {"[SEP]"};
        tokens.push_back(tokenizer.ConvertTokensToIds(sep_token)[0]);

        // 4. 验证长度（确保不超过context_length）
        if (tokens.size() > CONTEXT_LENGTH) {
            throw std::runtime_error("Token length exceeds context length");
        }

        for (int i = tokens.size(); i < CONTEXT_LENGTH; i++) {
            tokens.push_back(0);
        }
        ReadSingleToBuf(tokens, inDescs[0], inBufs[0], isDpico);
    } catch (const std::exception &e) {
        LOG(ERROR) << "error: " << e.what();
        return -1;
    }
    return 1;
}
} // namespace Infer