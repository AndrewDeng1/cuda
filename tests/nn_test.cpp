#include "../tensor.h"
#include "../nn.h"
#include <iostream>
#include <cassert>

using namespace std;

void test_linear() {
    cout << "=== Testing Linear ===" << endl;
    
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
    
    cout << "Linear test PASSED\n" << endl;
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

void test_layer_norm() {
    cout << "=== Testing LayerNorm ===" << endl;
    
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
    
    cout << "LayerNorm test PASSED\n" << endl;
}

void test_embedding() {
    cout << "=== Testing Embedding ===" << endl;
    
    Embedding emb(10, 4);  // vocab size 10, embedding dim 4
    
    // Create indices tensor
    Tensor indices({3}, {0, 5, 9});
    cout << "Indices: [0, 5, 9]" << endl;
    
    Tensor y = emb.forward(indices);
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 3 && y.shape()[1] == 4);
    
    cout << "Embedding test PASSED\n" << endl;
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

void test_sequential_with_make_module() {
    cout << "=== Testing Sequential with make_module ===" << endl;
    
    // Using vector constructor
    vector<unique_ptr<Module>> modules;
    modules.push_back(make_module<Linear>(4, 8));
    modules.push_back(make_module<Dropout>(0.1f));
    modules.push_back(make_module<Linear>(8, 2));
    
    Sequential seq(std::move(modules));
    seq.eval();  // Turn off dropout for deterministic test
    
    Tensor x({2, 4}, 1.0f);
    Tensor y = seq.forward(x);
    
    cout << "Output shape: [" << y.shape()[0] << ", " << y.shape()[1] << "]" << endl;
    assert(y.shape()[0] == 2 && y.shape()[1] == 2);
    
    cout << "Sequential with make_module test PASSED\n" << endl;
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

// Custom module example
class MLP : public Module {
public:
    Linear fc1_;
    Linear fc2_;
    Dropout drop_;
    
    MLP(int in_features, int hidden, int out_features)
        : fc1_(in_features, hidden),
          fc2_(hidden, out_features),
          drop_(0.1f) {
        register_module("fc1", &fc1_);
        register_module("fc2", &fc2_);
        register_module("drop", &drop_);
    }
    
    // Need custom move since we have Module members
    MLP(MLP&& other) noexcept
        : Module(std::move(other)),
          fc1_(std::move(other.fc1_)),
          fc2_(std::move(other.fc2_)),
          drop_(std::move(other.drop_)) {
        submodules_.clear();
        register_module("fc1", &fc1_);
        register_module("fc2", &fc2_);
        register_module("drop", &drop_);
    }
    
    Tensor forward(const Tensor& x) override {
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

void test_manual(){
    cout << "=== Testing Manual Module ===" << endl;

    
}

int main() {
    cout << "========================================" << endl;
    cout << "       Neural Network Module Tests     " << endl;
    cout << "========================================\n" << endl;
    
    test_linear();
    test_linear_no_bias();
    test_dropout();
    test_layer_norm();
    test_embedding();
    test_module_list();
    test_sequential();
    test_sequential_with_make_module();
    test_train_eval_mode();
    test_simple_forward();
    test_custom_module();
    
    cout << "========================================" << endl;
    cout << "       ALL TESTS PASSED!               " << endl;
    cout << "========================================" << endl;
    
    return 0;
}

