// Cross Entropy Test

#include "tensor.h"
#include <iostream>

int main() {
    // ===== BASIC 2D CROSS ENTROPY TEST =====
    cout << "===== BASIC 2D CROSS ENTROPY TEST =====" << endl;
    
    // Create logits (2 samples, 3 classes)
    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);

    // Create one-hot labels (class 2 for sample 1, class 1 for sample 2)
    Tensor labels({2, 3}, {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}, false);

    cout << "Logits:" << endl;
    logits.print();
    cout << "Labels (one-hot):" << endl;
    labels.print();

    // Compute cross entropy
    Tensor loss = logits.cross_entropy(labels, 1, true);
    cout << "Cross Entropy Loss:" << endl;
    loss.print();
    // Expected: CE = log(sum(exp(x))) - x[correct_class]
    // Sample 1: log(exp(1)+exp(2)+exp(3)) - 3 = log(2.718+7.389+20.086) - 3 = log(30.193) - 3 = 3.408 - 3 = 0.408
    // Sample 2: log(exp(4)+exp(5)+exp(6)) - 5 = log(54.598+148.413+403.429) - 5 = log(606.44) - 5 = 6.408 - 5 = 1.408

    // Backward pass
    loss.set_grad(Tensor({2, 1}, {1.0f, 1.0f}, true));
    loss.backward();
    cout << "Gradients for logits:" << endl;
    logits.grad().print();
    // Expected: softmax(logits) - labels

    // ===== LARGE VALUES TEST (NUMERICAL STABILITY) =====
    cout << "\n===== LARGE VALUES TEST (NUMERICAL STABILITY) =====" << endl;
    
    Tensor logits_large({1, 3}, {100.0f, 101.0f, 102.0f}, true);
    Tensor labels_large({1, 3}, {0.0f, 0.0f, 1.0f}, false);

    cout << "Logits (large values):" << endl;
    logits_large.print();
    cout << "Labels:" << endl;
    labels_large.print();

    Tensor loss_large = logits_large.cross_entropy(labels_large, 1, true);
    cout << "Cross Entropy Loss (should NOT be NaN):" << endl;
    loss_large.print();
    // Should be: log(exp(100)+exp(101)+exp(102)) - 102
    // = log(exp(100)*(1+e+e^2)) - 102
    // = 100 + log(1+e+e^2) - 102
    // = log(1+e+e^2) - 2
    // ≈ log(12.1) - 2 ≈ 2.49 - 2 = 0.408

    loss_large.set_grad(Tensor({1, 1}, {1.0f}, true));
    loss_large.backward();
    cout << "Gradients for large logits:" << endl;
    logits_large.grad().print();

    // ===== 3D CROSS ENTROPY TEST =====
    cout << "\n===== 3D CROSS ENTROPY TEST =====" << endl;
    
    // (batch=2, sequence=2, classes=3)
    Tensor logits_3d({2, 2, 3}, {
            // Batch 0, Seq 0
            1.0f, 2.0f, 3.0f,
            // Batch 0, Seq 1
            0.5f, 1.5f, 2.5f,
            // Batch 1, Seq 0
            2.0f, 1.0f, 0.0f,
            // Batch 1, Seq 1
            3.0f, 2.0f, 1.0f
    }, true);

    Tensor labels_3d({2, 2, 3}, {
            // Batch 0, Seq 0: class 2
            0.0f, 0.0f, 1.0f,
            // Batch 0, Seq 1: class 1
            0.0f, 1.0f, 0.0f,
            // Batch 1, Seq 0: class 0
            1.0f, 0.0f, 0.0f,
            // Batch 1, Seq 1: class 0
            1.0f, 0.0f, 0.0f
    }, false);

    cout << "3D Logits:" << endl;
    logits_3d.print();
    cout << "3D Labels:" << endl;
    labels_3d.print();

    Tensor loss_3d = logits_3d.cross_entropy(labels_3d, 2, true);
    cout << "3D Cross Entropy Loss:" << endl;
    loss_3d.print();

    loss_3d.set_grad(Tensor({2, 2, 1}, {1.0f, 1.0f, 1.0f, 1.0f}, true));
    loss_3d.backward();
    cout << "3D Gradients:" << endl;
    logits_3d.grad().print();

    // ===== NEGATIVE VALUES TEST =====
    cout << "\n===== NEGATIVE VALUES TEST =====" << endl;
    
    Tensor logits_neg({1, 3}, {-10.0f, -5.0f, -1.0f}, true);
    Tensor labels_neg({1, 3}, {1.0f, 0.0f, 0.0f}, false);

    cout << "Logits (negative values):" << endl;
    logits_neg.print();

    Tensor loss_neg = logits_neg.cross_entropy(labels_neg, 1, true);
    cout << "Cross Entropy Loss:" << endl;
    loss_neg.print();

    loss_neg.set_grad(Tensor({1, 1}, {1.0f}, true));
    loss_neg.backward();
    cout << "Gradients:" << endl;
    logits_neg.grad().print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
}

// ===== BASIC 2D CROSS ENTROPY TEST =====
// Logits:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// Labels (one-hot):
// Tensor(2, 3):
// [0 , 0 , 1 ], [0 , 1 , 0 ]
// Cross Entropy Loss:
// Tensor(2, 1):
// [0.407606 ], [1.40761 ]
// Gradients for logits:
// Tensor(2, 3):
// [0.0900306 , 0.244728 , -0.334759], [0.0900306 , -0.755271, 0.665241 ]

// ===== LARGE VALUES TEST (NUMERICAL STABILITY) =====
// Logits (large values):
// Tensor(1, 3):
// [100 , 101 , 102 ]
// Labels:
// Tensor(1, 3):
// [0 , 0 , 1 ]
// Cross Entropy Loss (should NOT be NaN):
// Tensor(1, 1):
// [0.407608 ]
// Gradients for large logits:
// Tensor(1, 3):
// [0.0900306 , 0.244728 , -0.334759]

// ===== 3D CROSS ENTROPY TEST =====
// 3D Logits:
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [0.5 , 1.5 , 2.5 ]], [[2 , 1 , 0 ], [3 , 2 , 1 ]]
// 3D Labels:
// Tensor(2, 2, 3):
// [[0 , 0 , 1 ], [0 , 1 , 0 ]], [[1 , 0 , 0 ], [1 , 0 , 0 ]]
// 3D Cross Entropy Loss:
// Tensor(2, 2, 1):
// [[0.407606 ], [1.40761 ]], [[0.407606 ], [0.407606 ]]
// 3D Gradients:
// Tensor(2, 2, 3):
// [[0.0900306 , 0.244728 , -0.334759], [0.0900306 , -0.755272, 0.665241 ]], [[-0.334759, 0.244728 , 0.0900306 ], [-0.334759, 0.244728 , 0.0900306 ]]

// ===== NEGATIVE VALUES TEST =====
// Logits (negative values):
// Tensor(1, 3):
// [-10, -5, -1]
// Cross Entropy Loss:
// Tensor(1, 1):
// [9.01827 ]
// Gradients:
// Tensor(1, 3):
// [-0.999879, 0.017984 , 0.981895 ]

// ===== ALL TESTS COMPLETE =====
