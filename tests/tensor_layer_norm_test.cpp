#include "tensor.h"
#include <iostream>
#include <cmath>

int main() {
    cout << "===== LAYER NORM BASIC 2D TEST =====" << endl;
    Tensor t1({2, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8
    }, true);
    Tensor gamma({4}, 1.0f, true);
    Tensor beta({4}, 0.0f, true);
    
    cout << "Input:" << endl;
    t1.print();
    cout << "Gamma (all 1s):" << endl;
    gamma.print();
    cout << "Beta (all 0s):" << endl;
    beta.print();
    
    Tensor ln1 = layer_norm(t1, gamma, beta);
    cout << "Layer norm output (should be normalized per row):" << endl;
    ln1.print();
    
    cout << "\n===== VERIFY NORMALIZATION =====" << endl;
    float sum1 = 0, sum2 = 0;
    for(int i = 0; i < 4; i++) {
        sum1 += ln1.at(i);
        sum2 += ln1.at(4 + i);
    }
    cout << "Row 0 mean (should be ~0): " << sum1 / 4 << endl;
    cout << "Row 1 mean (should be ~0): " << sum2 / 4 << endl;

    cout << "\n===== LAYER NORM WITH GAMMA/BETA =====" << endl;
    Tensor gamma2({4}, {2.0f, 2.0f, 2.0f, 2.0f}, true);
    Tensor beta2({4}, {1.0f, 1.0f, 1.0f, 1.0f}, true);
    
    cout << "Gamma (all 2s):" << endl;
    gamma2.print();
    cout << "Beta (all 1s):" << endl;
    beta2.print();
    
    Tensor ln2 = layer_norm(t1, gamma2, beta2);
    cout << "Layer norm with scale=2, shift=1:" << endl;
    ln2.print();

    cout << "\n===== LAYER NORM 3D TEST =====" << endl;
    Tensor t2({2, 3, 4}, {
        1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,
        13, 14, 15, 16,   17, 18, 19, 20,   21, 22, 23, 24
    }, true);
    Tensor gamma3({4}, 1.0f, true);
    Tensor beta3({4}, 0.0f, true);
    
    cout << "Input 2x3x4:" << endl;
    t2.print();
    
    Tensor ln3 = layer_norm(t2, gamma3, beta3);
    cout << "Layer norm (normalized over last dim):" << endl;
    ln3.print();

    cout << "\n===== LAYER NORM GRADIENT TEST =====" << endl;
    Tensor t3({2, 3}, {
        1, 2, 3,
        4, 5, 6
    }, true);
    Tensor gamma4({3}, {1.0f, 1.0f, 1.0f}, true);
    Tensor beta4({3}, {0.0f, 0.0f, 0.0f}, true);
    
    cout << "Input:" << endl;
    t3.print();
    
    Tensor ln4 = layer_norm(t3, gamma4, beta4);
    cout << "Layer norm output:" << endl;
    ln4.print();
    
    ln4.set_grad(Tensor({2, 3}, 1.0f, false));
    ln4.backward();
    
    cout << "Input gradient:" << endl;
    t3.grad().print();
    cout << "Gamma gradient:" << endl;
    gamma4.grad().print();
    cout << "Beta gradient:" << endl;
    beta4.grad().print();

    cout << "\n===== LAYER NORM SINGLE SAMPLE =====" << endl;
    Tensor t4({1, 4}, {0, 1, 2, 3}, true);
    Tensor gamma5({4}, 1.0f, true);
    Tensor beta5({4}, 0.0f, true);
    
    cout << "Input [0, 1, 2, 3]:" << endl;
    t4.print();
    
    Tensor ln5 = layer_norm(t4, gamma5, beta5);
    cout << "Layer norm output:" << endl;
    ln5.print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

// ===== LAYER NORM BASIC 2D TEST =====
// Input:
// Tensor(2, 4):
// [1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ]
// Gamma (all 1s):
// Tensor(4):
// 1 , 1 , 1 , 1
// Beta (all 0s):
// Tensor(4):
// 0 , 0 , 0 , 0
// Layer norm output (should be normalized per row):
// Tensor(2, 4):
// [-1.34164, -0.447214, 0.447214, 1.34164], [-1.34164, -0.447214, 0.447214, 1.34164]

// ===== VERIFY NORMALIZATION =====
// Row 0 mean (should be ~0): 0
// Row 1 mean (should be ~0): 0

// ===== LAYER NORM WITH GAMMA/BETA =====
// Gamma (all 2s):
// Tensor(4):
// 2 , 2 , 2 , 2
// Beta (all 1s):
// Tensor(4):
// 1 , 1 , 1 , 1
// Layer norm with scale=2, shift=1:
// Tensor(2, 4):
// [-1.68328, 0.105573, 1.89443, 3.68328], [-1.68328, 0.105573, 1.89443, 3.68328]

// ===== LAYER NORM 3D TEST =====
// Input 2x3x4:
// Tensor(2, 3, 4):
// [[1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ], [9 , 10 , 11 , 12 ]], [[13 , 14 , 15 , 16 ], [17 , 18 , 19 , 20 ], [21 , 22 , 23 , 24 ]]
// Layer norm (normalized over last dim):
// Tensor(2, 3, 4):
// [[-1.34164, -0.447214, 0.447214, 1.34164], [-1.34164, -0.447214, 0.447214, 1.34164], [-1.34164, -0.447214, 0.447214, 1.34164]], [[-1.34164, -0.447214, 0.447214, 1.34164], [-1.34164, -0.447214, 0.447214, 1.34164], [-1.34164, -0.447214, 0.447214, 1.34164]]

// ===== LAYER NORM GRADIENT TEST =====
// Input:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// Layer norm output:
// Tensor(2, 3):
// [-1.22474, 0, 1.22474], [-1.22474, 0, 1.22474]
// Input gradient:
// Tensor(2, 3):
// [0 , 0 , 0 ], [0 , 0 , 0 ]
// Gamma gradient:
// Tensor(3):
// -2.44949, 0, 2.44949
// Beta gradient:
// Tensor(3):
// 2, 2, 2

// ===== LAYER NORM SINGLE SAMPLE =====
// Input [0, 1, 2, 3]:
// Tensor(1, 4):
// [0 , 1 , 2 , 3 ]
// Layer norm output:
// Tensor(1, 4):
// [-1.34164, -0.447214, 0.447214, 1.34164]

// ===== ALL TESTS COMPLETE =====
