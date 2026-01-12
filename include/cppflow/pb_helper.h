//
// Created by Yury Lysogorskiy on 12.01.26.
//

#ifndef CPPFLOW_PB_HELPER_H
#define CPPFLOW_PB_HELPER_H
#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace cppflow {

struct TensorInfo {
    std::string name;
};

struct Signature {
    std::string key;
    std::map<std::string, TensorInfo> inputs;
    std::map<std::string, TensorInfo> outputs;
};

// A safe, read-only Protobuf wire-format reader
class ProtoReader {
    const uint8_t* ptr_;
    const uint8_t* end_;

public:
    explicit ProtoReader(const std::string& data)
        : ptr_(reinterpret_cast<const uint8_t*>(data.data())),
          end_(ptr_ + data.size()) {}

    ProtoReader(const uint8_t* ptr, size_t len)
        : ptr_(ptr), end_(ptr + len) {}

    bool eof() const { return ptr_ >= end_; }

    // Read Varint (Base-128)
    uint64_t read_varint() {
        uint64_t val = 0;
        int shift = 0;
        while (ptr_ < end_) {
            uint8_t b = *ptr_++;
            val |= static_cast<uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        return val;
    }

    // Read N bytes as a string (Raw Data)
    std::string read_bytes(size_t len) {
        if (ptr_ + len > end_) return ""; // Safety check
        std::string s(reinterpret_cast<const char*>(ptr_), len);
        ptr_ += len;
        return s;
    }

    // Read a Standard String Field (Length-Delimited)
    std::string read_string() {
        uint64_t len = read_varint();
        return read_bytes(len);
    }

    // Skip a field based on wire type
    void skip(uint32_t wire_type) {
        if (wire_type == 0) { // Varint
            read_varint();
        } else if (wire_type == 2) { // Length Delimited
            uint64_t len = read_varint();
            ptr_ += len;
        } else if (wire_type == 5) { // 32-bit
            ptr_ += 4;
        } else if (wire_type == 1) { // 64-bit
            ptr_ += 8;
        }
    }
};

// Helper: Parse a "TensorInfo" message to find field 1 (name)
inline TensorInfo ParseTensorInfo(const std::string& blob) {
    ProtoReader reader(blob);
    TensorInfo info;
    while (!reader.eof()) {
        uint64_t tag = reader.read_varint();
        uint32_t field = tag >> 3;
        if (field == 1) info.name = reader.read_string();
        else reader.skip(tag & 7);
    }
    return info;
}

// Helper: Parse the Inputs/Outputs Map Entry
// MapEntry<String, TensorInfo>
inline void ParseIOMapEntry(const std::string& blob, std::map<std::string, TensorInfo>& target_map) {
    ProtoReader reader(blob);
    std::string key;
    std::string value_blob; // Raw bytes of TensorInfo

    while (!reader.eof()) {
        uint64_t tag = reader.read_varint();
        uint32_t field = tag >> 3;
        if (field == 1) key = reader.read_string();
        else if (field == 2) value_blob = reader.read_string(); // Read raw bytes for lazy parsing
        else reader.skip(tag & 7);
    }

    if (!key.empty() && !value_blob.empty()) {
        target_map[key] = ParseTensorInfo(value_blob);
    }
}

// Helper: Parse a "SignatureDef" message
inline Signature ParseSignatureDef(const std::string& blob) {
    ProtoReader reader(blob);
    Signature sig;

    while (!reader.eof()) {
        uint64_t tag = reader.read_varint();
        uint32_t field = tag >> 3;

        // Field 1: inputs (Map<String, TensorInfo>)
        // Field 2: outputs (Map<String, TensorInfo>)
        if (field == 1 || field == 2) {
            std::string map_entry_blob = reader.read_string();
            if (field == 1) ParseIOMapEntry(map_entry_blob, sig.inputs);
            else ParseIOMapEntry(map_entry_blob, sig.outputs);
        } else {
            reader.skip(tag & 7);
        }
    }
    return sig;
}

// Main Function: Parse MetaGraphDef to find "signature_def" (Field 5)
inline std::map<std::string, Signature> ParseSignatures(const std::string& blob) {
    std::map<std::string, Signature> signatures;
    ProtoReader reader(blob);

    while (!reader.eof()) {
        uint64_t tag = reader.read_varint();
        uint32_t field = tag >> 3;

        // MetaGraphDef -> Field 5 is "signature_def" (Map<String, SignatureDef>)
        if (field == 5) {
            std::string map_entry_blob = reader.read_string();
            ProtoReader entry_reader(map_entry_blob);

            std::string sig_key;
            std::string sig_val_blob;

            while (!entry_reader.eof()) {
                uint64_t m_tag = entry_reader.read_varint();
                uint32_t m_field = m_tag >> 3;
                if (m_field == 1) sig_key = entry_reader.read_string();
                else if (m_field == 2) sig_val_blob = entry_reader.read_string();
                else entry_reader.skip(m_tag & 7);
            }

            if (!sig_key.empty() && !sig_val_blob.empty()) {
                signatures[sig_key] = ParseSignatureDef(sig_val_blob);
                signatures[sig_key].key = sig_key;
            }
        } else {
            reader.skip(tag & 7);
        }
    }
    return signatures;
}

} // namespace cppflow
#endif //CPPFLOW_PB_HELPER_H