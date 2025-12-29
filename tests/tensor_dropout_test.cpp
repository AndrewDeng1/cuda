#include "tensor.h"
#include <iostream>
#include <cmath>

int main() {
    cout << "===== DROPOUT BASIC TEST (p=0.5, training=true) =====" << endl;
    Tensor t1({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, true);
    cout << "Input:" << endl;
    t1.print();
    
    Tensor d1 = dropout(t1, 0.5f, true);
    cout << "After dropout (p=0.5, training=true):" << endl;
    d1.print();
    
    int zeros = 0;
    for(int i = 0; i < d1.size(); i++) {
        if(d1.at(i) == 0.0f) zeros++;
    }
    cout << "Zeros: " << zeros << " out of " << d1.size() << endl;

    cout << "\n===== DROPOUT INFERENCE TEST (training=false) =====" << endl;
    Tensor t2({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, true);
    cout << "Input:" << endl;
    t2.print();
    
    Tensor d2 = dropout(t2, 0.5f, false);
    cout << "After dropout (p=0.5, training=false) - should be unchanged:" << endl;
    d2.print();
    
    bool unchanged = true;
    for(int i = 0; i < t2.size(); i++) {
        if(abs(t2.at(i) - d2.at(i)) > 1e-5f) unchanged = false;
    }
    cout << "Output unchanged: " << (unchanged ? "YES" : "NO") << endl;

    cout << "\n===== DROPOUT SCALING TEST =====" << endl;
    Tensor t3({1, 4}, {2.0f, 2.0f, 2.0f, 2.0f}, true);
    cout << "Input (all 2.0):" << endl;
    t3.print();
    
    Tensor d3 = dropout(t3, 0.5f, true);
    cout << "After dropout (p=0.5) - non-zero values should be 4.0 (scaled by 1/(1-0.5)=2):" << endl;
    d3.print();
    
    bool correct_scale = true;
    for(int i = 0; i < d3.size(); i++) {
        if(d3.at(i) != 0.0f && abs(d3.at(i) - 4.0f) > 1e-5f) correct_scale = false;
    }
    cout << "Scaling correct: " << (correct_scale ? "YES" : "NO") << endl;

    cout << "\n===== DROPOUT GRADIENT TEST =====" << endl;
    Tensor t4({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    cout << "Input:" << endl;
    t4.print();
    
    Tensor d4 = dropout(t4, 0.3f, true);
    cout << "After dropout (p=0.3):" << endl;
    d4.print();
    
    d4.set_grad(Tensor({2, 3}, 1.0f, false));
    d4.backward();
    
    cout << "Input gradient (should match dropout pattern):" << endl;
    t4.grad().print();
    
    bool grad_matches = true;
    float scale = 1.0f / (1.0f - 0.3f);
    for(int i = 0; i < d4.size(); i++) {
        if(d4.at(i) == 0.0f && t4.grad().at(i) != 0.0f) grad_matches = false;
        if(d4.at(i) != 0.0f && abs(t4.grad().at(i) - scale) > 1e-5f) grad_matches = false;
    }
    cout << "Gradient pattern matches: " << (grad_matches ? "YES" : "NO") << endl;

    cout << "\n===== DROPOUT p=0 TEST =====" << endl;
    Tensor t5({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    Tensor d5 = dropout(t5, 0.0f, true);
    cout << "Input:" << endl;
    t5.print();
    cout << "After dropout (p=0) - should be unchanged:" << endl;
    d5.print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

// ===== DROPOUT BASIC TEST (p=0.5, training=true) =====
// Input:
// Tensor(2, 4):
// [1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ]
// After dropout (p=0.5, training=true):
// Tensor(2, 4):
// [2 , 4 , 0 , 0 ], [0 , 0 , 14 , 0 ]
// Zeros: 5 out of 8

// ===== DROPOUT INFERENCE TEST (training=false) =====
// Input:
// Tensor(2, 4):
// [1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ]
// After dropout (p=0.5, training=false) - should be unchanged:
// Tensor(2, 4):
// [1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ]
// Output unchanged: YES

// ===== DROPOUT SCALING TEST =====
// Input (all 2.0):
// Tensor(1, 4):
// [2 , 2 , 2 , 2 ]
// After dropout (p=0.5) - non-zero values should be 4.0 (scaled by 1/(1-0.5)=2):
// Tensor(1, 4):
// [0 , 0 , 0 , 0 ]
// Scaling correct: YES

// ===== DROPOUT GRADIENT TEST =====
// Input:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// After dropout (p=0.3):
// Tensor(2, 3):
// [1.42857 , 2.85714 , 0 ], [5.71429 , 7.14286 , 8.57143 ]
// Input gradient (should match dropout pattern):
// Tensor(2, 3):
// [1.42857 , 1.42857 , 0 ], [1.42857 , 1.42857 , 1.42857 ]
// Gradient pattern matches: YES

// ===== DROPOUT p=0 TEST =====
// Input:
// Tensor(2, 2):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// After dropout (p=0) - should be unchanged:
// Tensor(2, 2):
// [1 , 2 ], [3 , 4 ]

// ===== ALL TESTS COMPLETE =====
