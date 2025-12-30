#include "../tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

int main() {
    cout << "========================================" << endl;
    cout << "  Cross Entropy, Zeros, Ones Test       " << endl;
    cout << "========================================\n" << endl;
    
    // Test zeros and ones
    cout << "=== Testing zeros() and ones() ===" << endl;
    Tensor z = zeros({2, 3});
    Tensor o = ones({2, 3});
    
    cout << "Zeros tensor:" << endl;
    z.print();
    cout << "Ones tensor:" << endl;
    o.print();
    
    for(int i = 0; i < z.size(); i++) {
        assert(abs(z.at(i) - 0.0f) < 1e-6);
        assert(abs(o.at(i) - 1.0f) < 1e-6);
    }
    cout << "zeros() and ones() test PASSED\n" << endl;
    
    // Test cross_entropy with class indices (PyTorch style)
    cout << "=== Testing cross_entropy() with class indices ===" << endl;
    
    // Simple 2D case: (2 samples, 3 classes)
    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    Tensor targets({2}, {2.0f, 1.0f}, false);  // Class 2 for sample 0, class 1 for sample 1
    
    cout << "Logits:" << endl;
    logits.print();
    cout << "Targets (class indices):" << endl;
    targets.print();
    
    cout << "About to call cross_entropy..." << endl;
    Tensor loss = cross_entropy(logits, targets);
    cout << "cross_entropy returned successfully" << endl;
    
    cout << "Cross Entropy Loss (scalar):" << endl;
    loss.print();
    
    // Expected: 
    // Sample 0: -log(softmax([1,2,3])[2]) = -log(exp(3)/(exp(1)+exp(2)+exp(3))) 
    //          = -log(20.086/30.193) = -log(0.665) ≈ 0.408
    // Sample 1: -log(softmax([4,5,6])[1]) = -log(exp(5)/(exp(4)+exp(5)+exp(6)))
    //          = -log(148.413/606.44) = -log(0.245) ≈ 1.408
    // Mean: (0.408 + 1.408) / 2 ≈ 0.908
    
    assert(loss.shape().size() == 1 && loss.shape()[0] == 1);  // Scalar
    assert(loss.at(0) > 0.0f);
    
    cout << "Loss value: " << loss.at(0) << endl;
    
    // Test backward pass
    loss.set_grad(ones({1}));
    loss.backward();
    
    cout << "\nLogits gradients:" << endl;
    logits.grad().print();
    
    // Gradients should be (softmax - one_hot) / num_samples
    // For sample 0: softmax([1,2,3]) = [0.090, 0.245, 0.665], one_hot = [0,0,1]
    //              grad = [0.090, 0.245, -0.335] / 2 = [0.045, 0.122, -0.167]
    // For sample 1: softmax([4,5,6]) = [0.090, 0.245, 0.665], one_hot = [0,1,0]
    //              grad = [0.090, -0.755, 0.665] / 2 = [0.045, -0.377, 0.332]
    
    assert(logits.has_grad());
    cout << "Backward pass test PASSED\n" << endl;
    
    // Test 3D case (like GPT: B*T, vocab_size)
    cout << "=== Testing cross_entropy() with 3D tensors ===" << endl;
    
    Tensor logits_3d({2, 2, 3}, {
        1.0f, 2.0f, 3.0f,  // Batch 0, Seq 0
        0.5f, 1.5f, 2.5f,  // Batch 0, Seq 1
        2.0f, 1.0f, 0.0f,  // Batch 1, Seq 0
        3.0f, 2.0f, 1.0f   // Batch 1, Seq 1
    }, true);
    
    Tensor targets_3d({2, 2}, {
        2.0f, 1.0f,  // Batch 0: class 2, class 1
        0.0f, 0.0f   // Batch 1: class 0, class 0
    }, false);
    
    Tensor loss_3d = cross_entropy(logits_3d, targets_3d);
    
    cout << "3D Cross Entropy Loss:" << endl;
    loss_3d.print();
    
    assert(loss_3d.shape().size() == 1 && loss_3d.shape()[0] == 1);  // Scalar
    assert(loss_3d.at(0) > 0.0f);
    
    cout << "3D test PASSED\n" << endl;
    
    cout << "========================================" << endl;
    cout << "       ALL TESTS PASSED!                " << endl;
    cout << "========================================" << endl;
    
    return 0;
}

// ========================================
//   Cross Entropy, Zeros, Ones Test       
// ========================================

// === Testing zeros() and ones() ===
// Zeros tensor:
// Tensor(2, 3):
// [0 , 0 , 0 ], [0 , 0 , 0 ]
// Ones tensor:
// Tensor(2, 3):
// [1 , 1 , 1 ], [1 , 1 , 1 ]
// zeros() and ones() test PASSED

// === Testing cross_entropy() with class indices ===
// Logits:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// Targets (class indices):
// Tensor(2):
// 2 , 1 
// About to call cross_entropy...
// cross_entropy returned successfully
// Cross Entropy Loss (scalar):
// Tensor(1):
// 0.907606 
// Loss value: 0.907606

// Logits gradients:
// Tensor(2, 3):
// [0.0450153 , 0.122364 , -0.167379], [0.0450153 , -0.377636, 0.33262 ]
// Backward pass test PASSED

// === Testing cross_entropy() with 3D tensors ===
// 3D Cross Entropy Loss:
// Tensor(1):
// 0.657606 
// 3D test PASSED

// ========================================
//        ALL TESTS PASSED!                
// ========================================