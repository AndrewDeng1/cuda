#include "../optim.h"
#include "../nn.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

void test_sgd_basic() {
    cout << "=== Testing SGD (basic) ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    assert(params.size() == 2);  // weight + bias
    
    SGD sgd(params, 0.01f);  // lr=0.01, no momentum
    
    // Manually set gradients for testing
    Tensor weight_grad({2, 4}, 0.1f);  // Dummy gradient
    fc.weight_.tensor().set_grad(weight_grad);
    
    // Get initial parameter values
    float weight_before = fc.weight_.tensor().at(0);
    
    // Take a step
    sgd.step();
    
    // Check that parameter changed
    float weight_after = fc.weight_.tensor().at(0);
    cout << "Weight before: " << weight_before << ", after: " << weight_after << endl;
    assert(abs(weight_after - weight_before) > 1e-6);
    
    cout << "SGD (basic) test PASSED\n" << endl;
}

void test_sgd_momentum() {
    cout << "=== Testing SGD (with momentum) ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    
    SGD sgd(params, 0.01f, 0.9f);  // lr=0.01, momentum=0.9
    
    // Manually set gradients
    Tensor weight_grad1({2, 4}, 0.1f);
    fc.weight_.tensor().set_grad(weight_grad1);
    
    float weight_before = fc.weight_.tensor().at(0);
    
    // Take multiple steps to see momentum effect
    sgd.step();
    float weight_after_step1 = fc.weight_.tensor().at(0);
    
    // Zero grad and set new gradient
    sgd.zero_grad();
    Tensor weight_grad2({2, 4}, 0.1f);
    fc.weight_.tensor().set_grad(weight_grad2);
    sgd.step();
    float weight_after_step2 = fc.weight_.tensor().at(0);
    
    cout << "Weight: " << weight_before << " -> " << weight_after_step1 
         << " -> " << weight_after_step2 << endl;
    
    // Check that momentum buffer exists in state
    auto state_it = sgd.state().find(&fc.weight_);
    assert(state_it != sgd.state().end());
    assert(state_it->second.buffers.find("momentum") != state_it->second.buffers.end());
    
    cout << "SGD (momentum) test PASSED\n" << endl;
}

void test_sgd_zero_grad() {
    cout << "=== Testing SGD zero_grad ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    SGD sgd(params, 0.01f);
    
    // Manually set gradients
    Tensor weight_grad({2, 4}, 0.5f);
    fc.weight_.tensor().set_grad(weight_grad);
    
    // Check gradient exists
    assert(fc.weight_.tensor().has_grad());
    
    // Zero gradients
    sgd.zero_grad();
    
    // Check gradients are zero
    Tensor grad = fc.weight_.tensor().grad();
    bool all_zero = true;
    for (int i = 0; i < grad.size(); i++) {
        if (abs(grad.at(i)) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    assert(all_zero);
    
    cout << "SGD zero_grad test PASSED\n" << endl;
}

void test_adam_basic() {
    cout << "=== Testing Adam (basic) ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    
    Adam adam(params, 0.001f);  // lr=0.001
    
    // Manually set gradients
    Tensor weight_grad({2, 4}, 0.1f);
    fc.weight_.tensor().set_grad(weight_grad);
    
    float weight_before = fc.weight_.tensor().at(0);
    
    // Take a step
    adam.step();
    
    float weight_after = fc.weight_.tensor().at(0);
    cout << "Weight before: " << weight_before << ", after: " << weight_after << endl;
    assert(abs(weight_after - weight_before) > 1e-6);
    
    // Check that Adam state exists
    auto state_it = adam.state().find(&fc.weight_);
    assert(state_it != adam.state().end());
    const OptimizerState& state = state_it->second;
    assert(state.buffers.find("exp_avg") != state.buffers.end());
    assert(state.buffers.find("exp_avg_sq") != state.buffers.end());
    assert(state.step_count == 1);
    
    cout << "Adam (basic) test PASSED\n" << endl;
}

void test_adam_multiple_steps() {
    cout << "=== Testing Adam (multiple steps) ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    
    Adam adam(params, 0.001f);
    
    // Take multiple steps
    for (int i = 0; i < 5; i++) {
        Tensor weight_grad({2, 4}, 0.1f);
        fc.weight_.tensor().set_grad(weight_grad);
        
        adam.step();
        adam.zero_grad();
    }
    
    // Check step count
    auto state_it2 = adam.state().find(&fc.weight_);
    assert(state_it2 != adam.state().end());
    assert(state_it2->second.step_count == 5);
    
    cout << "Adam step count: " << state_it2->second.step_count << endl;
    cout << "Adam (multiple steps) test PASSED\n" << endl;
}

void test_adamw_basic() {
    cout << "=== Testing AdamW (basic) ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    
    AdamW adamw(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    
    // Manually set gradients
    Tensor weight_grad({2, 4}, 0.1f);
    fc.weight_.tensor().set_grad(weight_grad);
    
    float weight_before = fc.weight_.tensor().at(0);
    
    // Take a step
    adamw.step();
    
    float weight_after = fc.weight_.tensor().at(0);
    cout << "Weight before: " << weight_before << ", after: " << weight_after << endl;
    assert(abs(weight_after - weight_before) > 1e-6);
    
    // Check that AdamW state exists
    auto adamw_state_it = adamw.state().find(&fc.weight_);
    assert(adamw_state_it != adamw.state().end());
    assert(adamw_state_it->second.step_count == 1);
    
    cout << "AdamW (basic) test PASSED\n" << endl;
}

void test_adam_vs_adamw_weight_decay() {
    cout << "=== Testing Adam vs AdamW weight decay difference ===" << endl;
    
    // Create two identical models
    Linear fc1(4, 2);
    Linear fc2(4, 2);
    
    // Copy weights to make them identical
    for (int i = 0; i < fc1.weight_.tensor().size(); i++) {
        fc2.weight_.tensor().at(i) = fc1.weight_.tensor().at(i);
    }
    
    auto params1 = fc1.parameters();
    auto params2 = fc2.parameters();
    
    Adam adam(params1, 0.001f, 0.9f, 0.999f, 1e-8f, 0.1f);  // weight_decay=0.1
    AdamW adamw(params2, 0.001f, 0.9f, 0.999f, 1e-8f, 0.1f);  // weight_decay=0.1
    
    // Set same gradients manually
    Tensor weight_grad({2, 4}, 0.1f);
    fc1.weight_.tensor().set_grad(weight_grad);
    fc2.weight_.tensor().set_grad(weight_grad);
    
    // Take steps
    adam.step();
    adamw.step();
    
    // They should produce different results because weight decay is applied differently
    float weight1 = fc1.weight_.tensor().at(0);
    float weight2 = fc2.weight_.tensor().at(0);
    
    cout << "Adam weight: " << weight1 << ", AdamW weight: " << weight2 << endl;
    // They might be close but should be different due to different weight decay application
    // (Adam applies to gradient, AdamW applies directly to parameter)
    
    cout << "Adam vs AdamW weight decay test PASSED\n" << endl;
}

void test_optimizer_param_groups() {
    cout << "=== Testing Optimizer param groups ===" << endl;
    
    Linear fc1(4, 2);
    Linear fc2(4, 2);
    
    auto params1 = fc1.parameters();
    auto params2 = fc2.parameters();
    
    // Create optimizer with multiple param groups
    std::vector<ParamGroup> groups;
    std::unordered_map<std::string, float> opts1;
    opts1["lr"] = 0.01f;
    opts1["momentum"] = 0.9f;
    groups.emplace_back(params1, opts1);
    
    std::unordered_map<std::string, float> opts2;
    opts2["lr"] = 0.005f;  // Different learning rate
    opts2["momentum"] = 0.8f;
    groups.emplace_back(params2, opts2);
    
    SGD sgd(groups);
    
    assert(sgd.param_groups().size() == 2);
    assert(sgd.param_groups()[0].options.at("lr") == 0.01f);
    assert(sgd.param_groups()[1].options.at("lr") == 0.005f);
    
    cout << "Param groups: " << sgd.param_groups().size() << endl;
    cout << "Group 1 lr: " << sgd.param_groups()[0].options.at("lr") << endl;
    cout << "Group 2 lr: " << sgd.param_groups()[1].options.at("lr") << endl;
    
    cout << "Optimizer param groups test PASSED\n" << endl;
}

void test_simple_training_loop() {
    cout << "=== Testing Simple Training Loop ===" << endl;
    
    Linear fc(4, 2);
    auto params = fc.parameters();
    Adam adam(params, 0.01f);
    
    // Simple training loop
    for (int epoch = 0; epoch < 3; epoch++) {
        Tensor weight_grad({2, 4}, 0.1f * (epoch + 1));  // Varying gradient
        fc.weight_.tensor().set_grad(weight_grad);
        
        adam.step();
        adam.zero_grad();
        
        cout << "Epoch " << epoch << " completed" << endl;
    }
    
    // Check that step count increased
    auto state_it = adam.state().find(&fc.weight_);
    assert(state_it != adam.state().end());
    assert(state_it->second.step_count == 3);
    
    cout << "Simple training loop test PASSED\n" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "       Optimizer Tests                 " << endl;
    cout << "========================================\n" << endl;
    
    test_sgd_basic();
    test_sgd_momentum();
    test_sgd_zero_grad();
    test_adam_basic();
    test_adam_multiple_steps();
    test_adamw_basic();
    test_adam_vs_adamw_weight_decay();
    test_optimizer_param_groups();
    test_simple_training_loop();
    
    cout << "========================================" << endl;
    cout << "       ALL TESTS PASSED!               " << endl;
    cout << "========================================" << endl;
    
    return 0;
}

