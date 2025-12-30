#include "../nn.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

int main() {
    cout << "========================================" << endl;
    cout << "       ReLU Module Test                 " << endl;
    cout << "========================================\n" << endl;
    
    // Create ReLU module
    ReLU relu;
    
    // Test with positive and negative values
    Tensor x({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f}, true);
    
    cout << "Input tensor:" << endl;
    x.print();
    
    // Forward pass
    Tensor y = relu.forward(x);
    
    cout << "\nOutput tensor (after ReLU):" << endl;
    y.print();
    
    // Verify results
    assert(abs(y.at(0) - 0.0f) < 1e-6);  // -2 -> 0
    assert(abs(y.at(1) - 0.0f) < 1e-6);  // -1 -> 0
    assert(abs(y.at(2) - 0.0f) < 1e-6);  // 0 -> 0
    assert(abs(y.at(3) - 1.0f) < 1e-6);   // 1 -> 1
    assert(abs(y.at(4) - 2.0f) < 1e-6);  // 2 -> 2
    assert(abs(y.at(5) - 3.0f) < 1e-6);  // 3 -> 3
    
    cout << "All assertions passed!" << endl;
    
    // Test backward pass
    cout << "\nTesting backward pass..." << endl;
    Tensor grad({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false);
    y.set_grad(grad);
    y.backward();
    
    cout << "Input gradients:" << endl;
    x.grad().print();
    
    // Verify gradients: should be 0 for negative inputs, 1 for positive inputs
    assert(abs(x.grad().at(0) - 0.0f) < 1e-6);  // -2 -> 0
    assert(abs(x.grad().at(1) - 0.0f) < 1e-6);  // -1 -> 0
    assert(abs(x.grad().at(2) - 0.0f) < 1e-6);  // 0 -> 0 (or could be 1, depends on implementation)
    assert(abs(x.grad().at(3) - 1.0f) < 1e-6);  // 1 -> 1
    assert(abs(x.grad().at(4) - 1.0f) < 1e-6);  // 2 -> 1
    assert(abs(x.grad().at(5) - 1.0f) < 1e-6);  // 3 -> 1
    
    cout << "Backward pass test passed!" << endl;
    
    // Test in Sequential
    cout << "\nTesting ReLU in Sequential..." << endl;
    Linear fc(4, 2);
    Sequential seq;
    seq.append(std::move(fc));
    seq.append(ReLU());
    
    Tensor input({2, 4}, 1.0f, true);
    Tensor output = seq.forward(input);
    
    cout << "Sequential output shape: [";
    for (int i = 0; i < output.shape().size(); i++) {
        cout << output.shape()[i];
        if (i < output.shape().size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    // Verify output is non-negative (ReLU applied)
    for (int i = 0; i < output.size(); i++) {
        assert(output.at(i) >= 0.0f);
    }
    
    cout << "Sequential test passed!" << endl;
    
    cout << "\n========================================" << endl;
    cout << "       ALL TESTS PASSED!                " << endl;
    cout << "========================================" << endl;
    
    return 0;
}

