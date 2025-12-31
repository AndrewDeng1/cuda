#include "nn.h"

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
