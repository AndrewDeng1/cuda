#include "../tensor.h"
#include "../nn.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

void test_linear_small() {
    cout << "=== Testing Linear (small) ===" << endl;
    
    Linear fc(4, 3);  // 4 inputs, 3 outputs
    
    Tensor x({2, 4}, 1.0f);  // batch of 2, 4 features each
    Tensor y = fc.forward(x);
    
    cout << "Input shape: [2, 4]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 2 && y.shape()[1] == 3);
    
    // Test parameters
    auto params = fc.parameters();
    cout << "Number of parameters: " << params.size() << " (weight + bias)" << endl;
    assert(params.size() == 2);
    
    cout << "Linear (small) test PASSED\n" << endl;
}

void test_linear_medium() {
    cout << "=== Testing Linear (medium) ===" << endl;
    
    Linear fc(64, 32);  // 64 inputs, 32 outputs
    
    Tensor x({16, 64}, 0.5f);  // batch of 16
    Tensor y = fc.forward(x);
    
    cout << "Input shape: [16, 64]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 16 && y.shape()[1] == 32);
    
    cout << "Linear (medium) test PASSED\n" << endl;
}

void test_linear_large() {
    cout << "=== Testing Linear (large) ===" << endl;
    
    Linear fc(256, 128);  // 256 inputs, 128 outputs
    
    Tensor x({32, 256}, 0.1f);  // batch of 32
    Tensor y = fc.forward(x);
    
    cout << "Input shape: [32, 256]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 32 && y.shape()[1] == 128);
    
    cout << "Linear (large) test PASSED\n" << endl;
}

void test_linear_no_bias() {
    cout << "=== Testing Linear (no bias) ===" << endl;
    
    Linear fc(4, 3, false);  // no bias
    
    Tensor x({2, 4}, 1.0f);
    Tensor y = fc.forward(x);
    
    auto params = fc.parameters();
    cout << "Number of parameters: " << params.size() << " (weight only)" << endl;
    assert(params.size() == 1);
    
    cout << "Linear (no bias) test PASSED\n" << endl;
}

void test_dropout() {
    cout << "=== Testing Dropout ===" << endl;
    
    Dropout drop(0.5f);
    
    Tensor x({4, 4}, 1.0f);
    
    // Training mode - should have some zeros
    drop.train();
    Tensor y_train = drop.forward(x);
    cout << "Training mode output (some values scaled, some zeroed):" << endl;
    y_train.print();
    
    // Eval mode - should be identity
    drop.eval();
    Tensor y_eval = drop.forward(x);
    cout << "Eval mode output (should be all 1s):" << endl;
    y_eval.print();
    
    // Check eval mode preserves values
    bool all_ones = true;
    for (int i = 0; i < y_eval.size(); i++) {
        if (abs(y_eval.at(i) - 1.0f) > 1e-5) all_ones = false;
    }
    assert(all_ones);
    
    cout << "Dropout test PASSED\n" << endl;
}

void test_dropout_large() {
    cout << "=== Testing Dropout (large) ===" << endl;
    
    Dropout drop(0.3f);
    
    Tensor x({64, 128}, 1.0f);
    
    // Training mode
    drop.train();
    Tensor y_train = drop.forward(x);
    
    // Count zeros
    int zero_count = 0;
    for (int i = 0; i < y_train.size(); i++) {
        if (abs(y_train.at(i)) < 1e-6) zero_count++;
    }
    float dropout_rate = (float)zero_count / y_train.size();
    cout << "Observed dropout rate: " << dropout_rate << " (expected ~0.3)" << endl;
    assert(dropout_rate > 0.15 && dropout_rate < 0.45);  // Allow some variance
    
    // Eval mode
    drop.eval();
    Tensor y_eval = drop.forward(x);
    bool all_ones = true;
    for (int i = 0; i < y_eval.size(); i++) {
        if (abs(y_eval.at(i) - 1.0f) > 1e-5) all_ones = false;
    }
    assert(all_ones);
    
    cout << "Dropout (large) test PASSED\n" << endl;
}

void test_layer_norm_small() {
    cout << "=== Testing LayerNorm (small) ===" << endl;
    
    LayerNorm ln(4);  // normalize over last dim of size 4
    
    Tensor x({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    cout << "Input:" << endl;
    x.print();
    
    Tensor y = ln.forward(x);
    cout << "Output (normalized):" << endl;
    y.print();
    
    // Check parameters
    auto params = ln.parameters();
    cout << "Number of parameters: " << params.size() << " (gamma + beta)" << endl;
    assert(params.size() == 2);
    
    cout << "LayerNorm (small) test PASSED\n" << endl;
}

void test_layer_norm_large() {
    cout << "=== Testing LayerNorm (large) ===" << endl;
    
    LayerNorm ln(256);
    
    Tensor x = randn({32, 256});
    cout << "Input shape: [32, 256]" << endl;
    
    Tensor y = ln.forward(x);
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 32 && y.shape()[1] == 256);
    
    // Verify output is normalized (mean ~0, std ~1 per row)
    // Just check first row
    float sum = 0, sum_sq = 0;
    for (int i = 0; i < 256; i++) {
        float val = y.at({0, i});
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / 256;
    float var = sum_sq / 256 - mean * mean;
    cout << "First row mean: " << mean << " (should be ~0)" << endl;
    cout << "First row var: " << var << " (should be ~1)" << endl;
    assert(abs(mean) < 0.1 && abs(var - 1.0) < 0.2);
    
    cout << "LayerNorm (large) test PASSED\n" << endl;
}

void test_embedding_small() {
    cout << "=== Testing Embedding (small) ===" << endl;
    
    Embedding emb(10, 4);  // vocab size 10, embedding dim 4
    
    // Create indices tensor
    Tensor indices({3}, {0, 5, 9});
    cout << "Indices: [0, 5, 9]" << endl;
    
    Tensor y = emb.forward(indices);
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 3 && y.shape()[1] == 4);
    
    cout << "Embedding (small) test PASSED\n" << endl;
}

void test_embedding_large() {
    cout << "=== Testing Embedding (large) ===" << endl;
    
    Embedding emb(1000, 128);  // vocab size 1000, embedding dim 128
    
    // Create batch of indices
    Tensor indices({32, 16}, 0.0f);  // batch of 32, seq len 16
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            indices.at({i, j}) = (float)((i * 16 + j) % 1000);
        }
    }
    
    Tensor y = emb.forward(indices);
    cout << "Input shape: [32, 16]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << ", " << y.shape()[2] << "]" << endl;
    assert(y.shape()[0] == 32 && y.shape()[1] == 16 && y.shape()[2] == 128);
    
    cout << "Embedding (large) test PASSED\n" << endl;
}

void test_module_list() {
    cout << "=== Testing ModuleList ===" << endl;
    
    ModuleList layers;
    layers.append(Linear(4, 8));
    layers.append(Linear(8, 4));
    layers.append(Linear(4, 2));
    
    cout << "ModuleList size: " << layers.size() << endl;
    assert(layers.size() == 3);
    
    // Forward through each manually
    Tensor x({2, 4}, 1.0f);
    Tensor h = x;
    for (size_t i = 0; i < layers.size(); i++) {
        h = layers[i]->forward(h);
        cout << "After layer " << i << " shape: [" << h.shape()[0] << ", " << h.shape()[1] << "]" << endl;
    }
    assert(h.shape()[0] == 2 && h.shape()[1] == 2);
    
    // Check parameter collection
    auto params = layers.parameters();
    cout << "Total parameters: " << params.size() << endl;
    assert(params.size() == 6);  // 3 layers * 2 params each
    
    cout << "ModuleList test PASSED\n" << endl;
}

void test_module_list_large() {
    cout << "=== Testing ModuleList (large) ===" << endl;
    
    ModuleList layers;
    layers.append(Linear(128, 256));
    layers.append(Linear(256, 256));
    layers.append(Linear(256, 128));
    layers.append(Linear(128, 64));
    
    cout << "ModuleList size: " << layers.size() << endl;
    assert(layers.size() == 4);
    
    Tensor x({16, 128}, 0.5f);
    Tensor h = x;
    for (size_t i = 0; i < layers.size(); i++) {
        h = layers[i]->forward(h);
    }
    cout << "Final output shape: [" << h.shape()[0] << ", " << h.shape()[1] << "]" << endl;
    assert(h.shape()[0] == 16 && h.shape()[1] == 64);
    
    auto params = layers.parameters();
    cout << "Total parameters: " << params.size() << endl;
    assert(params.size() == 8);  // 4 layers * 2 params each
    
    cout << "ModuleList (large) test PASSED\n" << endl;
}

void test_sequential() {
    cout << "=== Testing Sequential ===" << endl;
    
    Sequential seq;
    seq.append(Linear(4, 8));
    seq.append(Linear(8, 4));
    seq.append(Linear(4, 2));
    
    Tensor x({2, 4}, 1.0f);
    Tensor y = seq.forward(x);
    
    cout << "Input shape: [2, 4]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 2 && y.shape()[1] == 2);
    
    // Check parameter collection
    auto params = seq.parameters();
    cout << "Total parameters: " << params.size() << endl;
    assert(params.size() == 6);
    
    cout << "Sequential test PASSED\n" << endl;
}

void test_sequential_large() {
    cout << "=== Testing Sequential (large) ===" << endl;
    
    Sequential seq;
    seq.append(Linear(256, 512));
    seq.append(Linear(512, 256));
    seq.append(Linear(256, 128));
    
    Tensor x({64, 256}, 0.1f);
    Tensor y = seq.forward(x);
    
    cout << "Input shape: [64, 256]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 64 && y.shape()[1] == 128);
    
    cout << "Sequential (large) test PASSED\n" << endl;
}

void test_sequential_with_vector() {
    cout << "=== Testing Sequential with vector constructor ===" << endl;
    
    // Using vector constructor
    vector<unique_ptr<Module>> modules;
    modules.push_back(std::make_unique<Linear>(4, 8));
    modules.push_back(std::make_unique<Dropout>(0.1f));
    modules.push_back(std::make_unique<Linear>(8, 2));
    
    Sequential seq(std::move(modules));
    seq.eval();  // Turn off dropout for deterministic test
    
    Tensor x({2, 4}, 1.0f);
    Tensor y = seq.forward(x);
    
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 2 && y.shape()[1] == 2);
    
    cout << "Sequential with vector constructor test PASSED\n" << endl;
}

void test_train_eval_mode() {
    cout << "=== Testing train/eval mode propagation ===" << endl;
    
    Sequential seq;
    seq.append(Linear(4, 4));
    seq.append(Dropout(0.5f));
    seq.append(Linear(4, 2));
    
    // Check initial training mode
    assert(seq.is_training() == true);
    
    // Switch to eval
    seq.eval();
    assert(seq.is_training() == false);
    
    // Switch back to train
    seq.train();
    assert(seq.is_training() == true);
    
    cout << "Train/eval mode propagation test PASSED\n" << endl;
}

void test_simple_forward() {
    cout << "=== Testing Simple Network Forward ===" << endl;
    
    // Simple network: Linear -> output
    Linear fc(3, 2);
    
    Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
    
    // Forward
    Tensor y = fc.forward(x);
    cout << "Input shape: [2, 3]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    y.print();
    
    assert(y.shape()[0] == 2 && y.shape()[1] == 2);
    
    cout << "Simple network forward test PASSED\n" << endl;
}

// Custom module example - uses register_submodules() instead of manual move
class MLP : public Module {
public:
    Linear fc1_;
    Linear fc2_;
    Dropout drop_;
    
    MLP(int in_features, int hidden, int out_features)
        : fc1_(in_features, hidden),
          fc2_(hidden, out_features),
          drop_(0.1f) {
        // No need to register here - lazy registration handles it
    }
    
    // Override register_submodules to register our member modules
    void register_submodules() override {
        register_module("fc1", &fc1_);
        register_module("fc2", &fc2_);
        register_module("drop", &drop_);
    }
    
    Tensor forward(const Tensor& x) override {
        ensure_registered();
        Tensor h = fc1_.forward(x);
        h = relu(h);
        h = drop_.forward(h);
        h = fc2_.forward(h);
        return h;
    }
};

void test_custom_module() {
    cout << "=== Testing Custom Module (MLP) ===" << endl;
    
    MLP mlp(4, 8, 2);
    mlp.eval();
    
    Tensor x({2, 4}, 1.0f);
    Tensor y = mlp.forward(x);
    
    cout << "MLP output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 2 && y.shape()[1] == 2);
    
    auto params = mlp.parameters();
    cout << "MLP total parameters: " << params.size() << endl;
    assert(params.size() == 4);  // fc1 weight/bias + fc2 weight/bias
    
    cout << "Custom module test PASSED\n" << endl;
}

// Larger custom module
class DeepMLP : public Module {
public:
    Linear fc1_;
    Linear fc2_;
    Linear fc3_;
    Linear fc4_;
    LayerNorm ln1_;
    LayerNorm ln2_;
    Dropout drop_;
    
    DeepMLP(int in_features, int hidden, int out_features)
        : fc1_(in_features, hidden),
          fc2_(hidden, hidden),
          fc3_(hidden, hidden),
          fc4_(hidden, out_features),
          ln1_(hidden),
          ln2_(hidden),
          drop_(0.1f) {
    }
    
    void register_submodules() override {
        register_module("fc1", &fc1_);
        register_module("fc2", &fc2_);
        register_module("fc3", &fc3_);
        register_module("fc4", &fc4_);
        register_module("ln1", &ln1_);
        register_module("ln2", &ln2_);
        register_module("drop", &drop_);
    }
    
    Tensor forward(const Tensor& x) override {
        ensure_registered();
        
        Tensor h = fc1_.forward(x);
        h = relu(h);
        h = ln1_.forward(h);
        h = drop_.forward(h);
        
        h = fc2_.forward(h);
        h = relu(h);
        
        h = fc3_.forward(h);
        h = relu(h);
        h = ln2_.forward(h);
        h = drop_.forward(h);
        
        h = fc4_.forward(h);
        return h;
    }
};

void test_deep_custom_module() {
    cout << "=== Testing Deep Custom Module ===" << endl;
    
    DeepMLP model(128, 256, 64);
    model.eval();
    
    Tensor x({32, 128}, 0.5f);
    Tensor y = model.forward(x);
    
    cout << "Input shape: [32, 128]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 32 && y.shape()[1] == 64);
    
    auto params = model.parameters();
    cout << "Total parameters: " << params.size() << endl;
    // fc1: 2, fc2: 2, fc3: 2, fc4: 2, ln1: 2, ln2: 2 = 12
    assert(params.size() == 12);
    
    cout << "Deep custom module test PASSED\n" << endl;
}

// Test with embedding + linear
class EmbeddingNet : public Module {
public:
    Embedding emb_;
    Linear fc_;
    
    EmbeddingNet(int vocab_size, int embed_dim, int out_features)
        : emb_(vocab_size, embed_dim),
          fc_(embed_dim, out_features) {
    }
    
    void register_submodules() override {
        register_module("emb", &emb_);
        register_module("fc", &fc_);
    }
    
    Tensor forward(const Tensor& x) override {
        ensure_registered();
        // x: [batch, seq_len] indices
        Tensor h = emb_.forward(x);  // [batch, seq_len, embed_dim]
        // For simplicity, just take the first token's embedding
        // In practice you'd pool or use attention
        h = h.slice(1, 0, 1).reshape({(int)x.shape()[0], -1});  // [batch, embed_dim]
        h = fc_.forward(h);
        return h;
    }
};

void test_embedding_net() {
    cout << "=== Testing EmbeddingNet ===" << endl;
    
    EmbeddingNet net(100, 32, 10);  // vocab 100, embed dim 32, output 10
    
    // Create indices
    Tensor indices({4, 8}, 0.0f);  // batch 4, seq len 8
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            indices.at({i, j}) = (float)((i * 8 + j) % 100);
        }
    }
    
    Tensor y = net.forward(indices);
    cout << "Input shape: [4, 8]" << endl;
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 4 && y.shape()[1] == 10);
    
    auto params = net.parameters();
    cout << "Total parameters: " << params.size() << endl;
    assert(params.size() == 3);  // emb weight, fc weight, fc bias
    
    cout << "EmbeddingNet test PASSED\n" << endl;
}

void test_nested_modules() {
    cout << "=== Testing Nested Modules ===" << endl;
    
    // ModuleList containing Sequential modules
    ModuleList blocks;
    
    // Block 1: Linear -> LayerNorm
    Sequential block1;
    block1.append(Linear(64, 128));
    block1.append(LayerNorm(128));
    blocks.append(std::move(block1));
    
    // Block 2: Linear -> LayerNorm
    Sequential block2;
    block2.append(Linear(128, 64));
    block2.append(LayerNorm(64));
    blocks.append(std::move(block2));
    
    // Forward through all blocks
    Tensor x({8, 64}, 1.0f);
    Tensor h = x;
    for (size_t i = 0; i < blocks.size(); i++) {
        h = blocks[i]->forward(h);
    }
    
    cout << "Input shape: [8, 64]" << endl;
    cout << "Output shape: [" << h.shape()[0] << ", " << h.shape()[1] << "]" << endl;
    assert(h.shape()[0] == 8 && h.shape()[1] == 64);
    
    auto params = blocks.parameters();
    cout << "Total parameters: " << params.size() << endl;
    // Block1: Linear(2) + LayerNorm(2) = 4
    // Block2: Linear(2) + LayerNorm(2) = 4
    // Total = 8
    assert(params.size() == 8);
    
    cout << "Nested modules test PASSED\n" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "       Neural Network Module Tests     " << endl;
    cout << "========================================\n" << endl;
    
    // Small tests
    test_linear_small();
    test_linear_no_bias();
    test_dropout();
    test_layer_norm_small();
    test_embedding_small();
    test_module_list();
    test_sequential();
    test_sequential_with_vector();
    test_train_eval_mode();
    test_simple_forward();
    test_custom_module();
    
    // Medium/Large tests
    test_linear_medium();
    test_linear_large();
    test_dropout_large();
    test_layer_norm_large();
    test_embedding_large();
    test_module_list_large();
    test_sequential_large();
    test_deep_custom_module();
    test_embedding_net();
    test_nested_modules();
    
    cout << "========================================" << endl;
    cout << "       ALL TESTS PASSED!               " << endl;
    cout << "========================================" << endl;
    
    return 0;
}
