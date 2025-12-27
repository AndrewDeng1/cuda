// Cross Entropy Test

#include "tensor.h"
#include <iostream>

int main() {
    // ===== BASIC 2D CROSS ENTROPY TEST =====
    cout << "===== BASIC 2D CROSS ENTROPY TEST =====" << endl;
    
    // Create logits (2 samples, 3 classes)
    auto logits = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    // Create one-hot labels (class 2 for sample 1, class 1 for sample 2)
    auto labels = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
        false
    );

    cout << "Logits:" << endl;
    logits->print();
    cout << "Labels (one-hot):" << endl;
    labels->print();

    // Compute cross entropy
    auto loss = logits->cross_entropy(labels, 1, true);
    cout << "Cross Entropy Loss:" << endl;
    loss->print();
    // Expected: CE = log(sum(exp(x))) - x[correct_class]
    // Sample 1: log(exp(1)+exp(2)+exp(3)) - 3 = log(2.718+7.389+20.086) - 3 = log(30.193) - 3 = 3.408 - 3 = 0.408
    // Sample 2: log(exp(4)+exp(5)+exp(6)) - 5 = log(54.598+148.413+403.429) - 5 = log(606.44) - 5 = 6.408 - 5 = 1.408

    // Backward pass
    loss->grad = make_shared<Tensor>(vector<int>{2, 1}, vector<float>{1.0f, 1.0f}, true);
    loss->backward();
    cout << "Gradients for logits:" << endl;
    logits->grad->print();
    // Expected: softmax(logits) - labels

    // ===== LARGE VALUES TEST (NUMERICAL STABILITY) =====
    cout << "\n===== LARGE VALUES TEST (NUMERICAL STABILITY) =====" << endl;
    
    auto logits_large = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{100.0f, 101.0f, 102.0f},
        true
    );

    auto labels_large = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{0.0f, 0.0f, 1.0f},
        false
    );

    cout << "Logits (large values):" << endl;
    logits_large->print();
    cout << "Labels:" << endl;
    labels_large->print();

    auto loss_large = logits_large->cross_entropy(labels_large, 1, true);
    cout << "Cross Entropy Loss (should NOT be NaN):" << endl;
    loss_large->print();
    // Should be: log(exp(100)+exp(101)+exp(102)) - 102
    // = log(exp(100)*(1+e+e^2)) - 102
    // = 100 + log(1+e+e^2) - 102
    // = log(1+e+e^2) - 2
    // ≈ log(12.1) - 2 ≈ 2.49 - 2 = 0.408

    loss_large->grad = make_shared<Tensor>(vector<int>{1, 1}, vector<float>{1.0f}, true);
    loss_large->backward();
    cout << "Gradients for large logits:" << endl;
    logits_large->grad->print();

    // ===== 3D CROSS ENTROPY TEST =====
    cout << "\n===== 3D CROSS ENTROPY TEST =====" << endl;
    
    // (batch=2, sequence=2, classes=3)
    auto logits_3d = make_shared<Tensor>(
        vector<int>{2, 2, 3},
        vector<float>{
            // Batch 0, Seq 0
            1.0f, 2.0f, 3.0f,
            // Batch 0, Seq 1
            0.5f, 1.5f, 2.5f,
            // Batch 1, Seq 0
            2.0f, 1.0f, 0.0f,
            // Batch 1, Seq 1
            3.0f, 2.0f, 1.0f
        },
        true
    );

    auto labels_3d = make_shared<Tensor>(
        vector<int>{2, 2, 3},
        vector<float>{
            // Batch 0, Seq 0: class 2
            0.0f, 0.0f, 1.0f,
            // Batch 0, Seq 1: class 1
            0.0f, 1.0f, 0.0f,
            // Batch 1, Seq 0: class 0
            1.0f, 0.0f, 0.0f,
            // Batch 1, Seq 1: class 0
            1.0f, 0.0f, 0.0f
        },
        false
    );

    cout << "3D Logits:" << endl;
    logits_3d->print();
    cout << "3D Labels:" << endl;
    labels_3d->print();

    auto loss_3d = logits_3d->cross_entropy(labels_3d, 2, true);
    cout << "3D Cross Entropy Loss:" << endl;
    loss_3d->print();

    loss_3d->grad = make_shared<Tensor>(vector<int>{2, 2, 1}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, true);
    loss_3d->backward();
    cout << "3D Gradients:" << endl;
    logits_3d->grad->print();

    // ===== NEGATIVE VALUES TEST =====
    cout << "\n===== NEGATIVE VALUES TEST =====" << endl;
    
    auto logits_neg = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{-10.0f, -5.0f, -1.0f},
        true
    );

    auto labels_neg = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{1.0f, 0.0f, 0.0f},
        false
    );

    cout << "Logits (negative values):" << endl;
    logits_neg->print();

    auto loss_neg = logits_neg->cross_entropy(labels_neg, 1, true);
    cout << "Cross Entropy Loss:" << endl;
    loss_neg->print();

    loss_neg->grad = make_shared<Tensor>(vector<int>{1, 1}, vector<float>{1.0f}, true);
    loss_neg->backward();
    cout << "Gradients:" << endl;
    logits_neg->grad->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
}

