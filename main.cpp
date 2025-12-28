#include "tensor.h"
#include <iostream>
#include <cmath>

int main() {
    cout << "===== LAYER NORM BASIC 2D TEST =====" << endl;
    auto t1 = make_shared<Tensor>(vector<int>{2, 4}, vector<float>{
        1, 2, 3, 4,
        5, 6, 7, 8
    }, true);
    auto gamma = make_shared<Tensor>(vector<int>{4}, 1.0f, true);
    auto beta = make_shared<Tensor>(vector<int>{4}, 0.0f, true);
    
    cout << "Input:" << endl;
    t1->print();
    cout << "Gamma (all 1s):" << endl;
    gamma->print();
    cout << "Beta (all 0s):" << endl;
    beta->print();
    
    auto ln1 = layer_norm(t1, gamma, beta);
    cout << "Layer norm output (should be normalized per row):" << endl;
    ln1->print();
    
    cout << "\n===== VERIFY NORMALIZATION =====" << endl;
    float sum1 = 0, sum2 = 0;
    for(int i = 0; i < 4; i++) {
        sum1 += ln1->at(i);
        sum2 += ln1->at(4 + i);
    }
    cout << "Row 0 mean (should be ~0): " << sum1 / 4 << endl;
    cout << "Row 1 mean (should be ~0): " << sum2 / 4 << endl;

    cout << "\n===== LAYER NORM WITH GAMMA/BETA =====" << endl;
    auto gamma2 = make_shared<Tensor>(vector<int>{4}, vector<float>{2.0f, 2.0f, 2.0f, 2.0f}, true);
    auto beta2 = make_shared<Tensor>(vector<int>{4}, vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, true);
    
    cout << "Gamma (all 2s):" << endl;
    gamma2->print();
    cout << "Beta (all 1s):" << endl;
    beta2->print();
    
    auto ln2 = layer_norm(t1, gamma2, beta2);
    cout << "Layer norm with scale=2, shift=1:" << endl;
    ln2->print();

    cout << "\n===== LAYER NORM 3D TEST =====" << endl;
    auto t2 = make_shared<Tensor>(vector<int>{2, 3, 4}, vector<float>{
        1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,
        13, 14, 15, 16,   17, 18, 19, 20,   21, 22, 23, 24
    }, true);
    auto gamma3 = make_shared<Tensor>(vector<int>{4}, 1.0f, true);
    auto beta3 = make_shared<Tensor>(vector<int>{4}, 0.0f, true);
    
    cout << "Input 2x3x4:" << endl;
    t2->print();
    
    auto ln3 = layer_norm(t2, gamma3, beta3);
    cout << "Layer norm (normalized over last dim):" << endl;
    ln3->print();

    cout << "\n===== LAYER NORM GRADIENT TEST =====" << endl;
    auto t3 = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        1, 2, 3,
        4, 5, 6
    }, true);
    auto gamma4 = make_shared<Tensor>(vector<int>{3}, vector<float>{1.0f, 1.0f, 1.0f}, true);
    auto beta4 = make_shared<Tensor>(vector<int>{3}, vector<float>{0.0f, 0.0f, 0.0f}, true);
    
    cout << "Input:" << endl;
    t3->print();
    
    auto ln4 = layer_norm(t3, gamma4, beta4);
    cout << "Layer norm output:" << endl;
    ln4->print();
    
    ln4->grad = make_shared<Tensor>(vector<int>{2, 3}, 1.0f, false);
    ln4->backward();
    
    cout << "Input gradient:" << endl;
    t3->grad->print();
    cout << "Gamma gradient:" << endl;
    gamma4->grad->print();
    cout << "Beta gradient:" << endl;
    beta4->grad->print();

    cout << "\n===== LAYER NORM SINGLE SAMPLE =====" << endl;
    auto t4 = make_shared<Tensor>(vector<int>{1, 4}, vector<float>{0, 1, 2, 3}, true);
    auto gamma5 = make_shared<Tensor>(vector<int>{4}, 1.0f, true);
    auto beta5 = make_shared<Tensor>(vector<int>{4}, 0.0f, true);
    
    cout << "Input [0, 1, 2, 3]:" << endl;
    t4->print();
    
    auto ln5 = layer_norm(t4, gamma5, beta5);
    cout << "Layer norm output:" << endl;
    ln5->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}
