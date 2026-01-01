#include "nn.h"
#include <fstream>
#include <sstream>
#include <cstdint>
#include <stdexcept>
#include <regex>
#include <algorithm>

// ============================================================================
// Linear
// ============================================================================

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
    
    weight_ = Parameter(xavier_normal({out_features, in_features}));
    
    if (use_bias_) {
        bias_ = Parameter(Tensor({out_features}, 0.0f));
    }
    // No need to call register_params() - lazy registration handles it
}   

void Linear::register_params() {
    register_param("weight", &weight_);
    if (use_bias_) {
        register_param("bias", &bias_);
    }
}

Tensor Linear::forward(const Tensor& x) {
    // x: [..., in_features] -> [..., out_features]
    Tensor wT = weight_.tensor().transpose(-2, -1);
    Tensor out = matmul(x, wT);
    
    if (use_bias_) {
        out = out + bias_.tensor();
    }
    
    return out;
}

// ============================================================================
// Dropout
// ============================================================================

Dropout::Dropout(float p) : p_(p) {}

Tensor Dropout::forward(const Tensor& x) {
    return dropout(x, p_, training_);
}

// ============================================================================
// ReLU
// ============================================================================

Tensor ReLU::forward(const Tensor& x) {
    return relu(x);
}

// ============================================================================
// LayerNorm
// ============================================================================

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {
    
    gamma_ = Parameter(Tensor({normalized_shape}, 1.0f));
    beta_ = Parameter(Tensor({normalized_shape}, 0.0f));
}

void LayerNorm::register_params() {
    register_param("weight", &gamma_);
    register_param("bias", &beta_);
}

Tensor LayerNorm::forward(const Tensor& x) {
    return layer_norm(x, gamma_.tensor(), beta_.tensor(), eps_);
}

// ============================================================================
// Embedding
// ============================================================================

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    
    weight_ = Parameter(randn({num_embeddings, embedding_dim}));
}

void Embedding::register_params() {
    register_param("weight", &weight_);
}

Tensor Embedding::forward(const Tensor& indices) {
    return embedding(weight_.tensor(), indices);
}

// ============================================================================
// ModuleList
// ============================================================================

ModuleList::ModuleList(std::vector<std::unique_ptr<Module>> modules) {
    owned_modules_ = std::move(modules);
}

void ModuleList::append(std::unique_ptr<Module> m) {
    owned_modules_.push_back(std::move(m));
    registered_ = false;
}

void ModuleList::register_submodules() {
    for (size_t i = 0; i < owned_modules_.size(); ++i) {
        register_module(std::to_string(i), owned_modules_[i].get());
    }
}

// ============================================================================
// Sequential
// ============================================================================

Sequential::Sequential(std::vector<std::unique_ptr<Module>> modules) {
    owned_modules_ = std::move(modules);
}

void Sequential::append(std::unique_ptr<Module> m) {
    owned_modules_.push_back(std::move(m));
    registered_ = false;
}

void Sequential::register_submodules() {
    for (size_t i = 0; i < owned_modules_.size(); ++i) {
        register_module(std::to_string(i), owned_modules_[i].get());
    }
}

Tensor Sequential::forward(const Tensor& x) {
    ensure_registered();
    Tensor out = x;
    for (auto& m : owned_modules_) {
        out = m->forward(out);
    }
    return out;
}

// ============================================================================
// Module state_dict and load_state_dict
// ============================================================================

std::pair<std::vector<std::string>, std::vector<std::string>> Module::load_state_dict(
    const std::map<std::string, Tensor>& state_dict, bool strict) {
    
    ensure_registered();
    std::vector<std::string> missing_keys;
    std::vector<std::string> unexpected_keys;
    
    // Load recursively and get used keys
    std::set<std::string> used_keys = load_state_dict_recursive(state_dict, "", missing_keys, strict);
    
    // Find unexpected keys (keys in state_dict that weren't used)
    if (strict) {
        for (const auto& [key, _] : state_dict) {
            if (used_keys.find(key) == used_keys.end()) {
                unexpected_keys.push_back(key);
            }
        }
    }
    
    return std::make_pair(missing_keys, unexpected_keys);
}

// ============================================================================
// write_safe_tensors
// ============================================================================

void write_safe_tensors(const std::map<std::string, Tensor>& state_dict, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // Build header JSON
    std::ostringstream header_json;
    header_json << "{";
    
    // Calculate offsets and build tensor metadata
    size_t data_offset = 0;
    size_t idx = 0;
    
    for (const auto& [key, tensor] : state_dict) {
        if (idx > 0) header_json << ",";
        
        std::vector<int> shape = tensor.shape();
        int total_size = tensor.size();
        size_t data_size_bytes = total_size * sizeof(float);
        
        // Tensor entry in header
        header_json << "\"" << key << "\":{";
        header_json << "\"dtype\":\"F32\",";  // float32
        header_json << "\"shape\":[";
        for (size_t i = 0; i < shape.size(); i++) {
            header_json << shape[i];
            if (i < shape.size() - 1) header_json << ",";
        }
        header_json << "],";
        header_json << "\"data_offsets\":[" << data_offset << "," << (data_offset + data_size_bytes) << "]";
        header_json << "}";
        
        data_offset += data_size_bytes;
        idx++;
    }
    
    header_json << "}";
    
    std::string header_str = header_json.str();
    uint64_t header_len = header_str.size();
    
    // Write header length (8 bytes, little endian)
    file.write(reinterpret_cast<const char*>(&header_len), sizeof(uint64_t));
    
    // Write header JSON
    file.write(header_str.c_str(), header_len);
    
    // Write tensor data sequentially
    for (const auto& [key, tensor] : state_dict) {
        int total_size = tensor.size();
        float* data = tensor.data();
        file.write(reinterpret_cast<const char*>(data), total_size * sizeof(float));
    }
    
    file.close();
    std::cout << "Checkpoint saved to: " << filename << std::endl;
}

// ============================================================================
// load_safe_tensors
// ============================================================================

std::map<std::string, Tensor> load_safe_tensors(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // Read header length (8 bytes, little endian)
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(uint64_t));
    
    // Read header JSON
    std::vector<char> header_buf(header_len);
    file.read(header_buf.data(), header_len);
    std::string header_str(header_buf.data(), header_len);
    
    // Parse JSON to extract tensor metadata
    // Format: {"key1":{"dtype":"F32","shape":[1,2,3],"data_offsets":[0,24]},"key2":{...}}
    std::map<std::string, Tensor> state_dict;
    
    // Simple JSON parser for this specific format
    // Extract each tensor entry: "key":{"dtype":"F32","shape":[...],"data_offsets":[start,end]}
    std::string regex_pattern = "\"([^\"]+)\":\\{\"dtype\":\"F32\",\"shape\":\\[([^\\]]+)\\]"
                                ",\"data_offsets\":\\[(\\d+),(\\d+)\\]\\}";
    std::regex tensor_regex(regex_pattern);
    std::sregex_iterator iter(header_str.begin(), header_str.end(), tensor_regex);
    std::sregex_iterator end;
    
    // Calculate data start offset (8 bytes header_len + header_len bytes)
    size_t data_start = 8 + header_len;
    
    for (; iter != end; ++iter) {
        const std::smatch& match = *iter;
        std::string key = match[1].str();
        std::string shape_str = match[2].str();
        size_t offset_start = std::stoull(match[3].str());
        size_t offset_end = std::stoull(match[4].str());
        
        // Parse shape
        std::vector<int> shape;
        std::istringstream shape_stream(shape_str);
        std::string dim_str;
        while (std::getline(shape_stream, dim_str, ',')) {
            // Remove whitespace
            dim_str.erase(std::remove_if(dim_str.begin(), dim_str.end(), ::isspace), dim_str.end());
            if (!dim_str.empty()) {
                shape.push_back(std::stoi(dim_str));
            }
        }
        
        // Calculate size
        int total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        
        // Read tensor data
        size_t data_size_bytes = offset_end - offset_start;
        size_t file_pos = data_start + offset_start;
        file.seekg(file_pos, std::ios::beg);
        
        std::vector<float> data(total_size);
        file.read(reinterpret_cast<char*>(data.data()), data_size_bytes);
        
        // Create tensor
        Tensor tensor(shape, data, false);
        state_dict[key] = tensor;
    }
    
    file.close();
    std::cout << "Checkpoint loaded from: " << filename << " (" << state_dict.size() << " tensors)" << std::endl;
    
    return state_dict;
}
