#include "tensor.h"
#include <iostream>

int main() {
    cout << "===== EMBEDDING BASIC TEST =====" << endl;
    Tensor weight({5, 3}, {
        0, 1, 2,      // row 0
        10, 11, 12,   // row 1
        20, 21, 22,   // row 2
        30, 31, 32,   // row 3
        40, 41, 42    // row 4
    }, true);
    
    cout << "Embedding weight (5 x 3):" << endl;
    weight.print();
    
    vector<int> indices = {0, 2, 4};
    Tensor out1 = embedding(weight, indices);
    cout << "embedding(indices=[0, 2, 4]):" << endl;
    out1.print();

    cout << "\n===== EMBEDDING WITH REPEATS =====" << endl;
    vector<int> indices2 = {1, 1, 3, 1};
    Tensor out2 = embedding(weight, indices2);
    cout << "embedding(indices=[1, 1, 3, 1]):" << endl;
    out2.print();

    cout << "\n===== EMBEDDING SINGLE INDEX =====" << endl;
    vector<int> indices3 = {3};
    Tensor out3 = embedding(weight, indices3);
    cout << "embedding(indices=[3]):" << endl;
    out3.print();

    cout << "\n===== EMBEDDING GRADIENT TEST =====" << endl;
    Tensor weight2({4, 2}, {
        1, 2,    // row 0
        3, 4,    // row 1
        5, 6,    // row 2
        7, 8     // row 3
    }, true);
    
    cout << "Weight (4 x 2):" << endl;
    weight2.print();
    
    vector<int> indices4 = {0, 2, 3};
    Tensor out4 = embedding(weight2, indices4);
    cout << "embedding(indices=[0, 2, 3]):" << endl;
    out4.print();
    
    out4.set_grad(Tensor({3, 2}, {
        10, 20,   // grad for row 0
        30, 40,   // grad for row 2
        50, 60    // grad for row 3
    }, false));
    cout << "Upstream gradient:" << endl;
    out4.grad().print();
    
    out4.backward();
    cout << "Weight gradient (row 1 should be 0):" << endl;
    weight2.grad().print();

    cout << "\n===== EMBEDDING GRADIENT WITH REPEATS =====" << endl;
    Tensor weight3({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    }, true);
    
    cout << "Weight (3 x 2):" << endl;
    weight3.print();
    
    vector<int> indices5 = {0, 1, 0, 1, 0};  // 0 appears 3 times, 1 appears 2 times
    Tensor out5 = embedding(weight3, indices5);
    cout << "embedding(indices=[0, 1, 0, 1, 0]):" << endl;
    out5.print();
    
    out5.set_grad(Tensor({5, 2}, {
        1, 1,    // row 0 first
        2, 2,    // row 1 first
        3, 3,    // row 0 second
        4, 4,    // row 1 second
        5, 5     // row 0 third
    }, false));
    cout << "Upstream gradient:" << endl;
    out5.grad().print();
    
    out5.backward();
    cout << "Weight gradient (row 0: 1+3+5=9, row 1: 2+4=6, row 2: 0):" << endl;
    weight3.grad().print();

    cout << "\n===== EMBEDDING FOR SEQUENCE =====" << endl;
    Tensor token_emb({10, 4}, 0.0f, true);
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 4; j++) {
            token_emb.at(i * 4 + j) = i * 10 + j;
        }
    }
    cout << "Token embedding table (vocab=10, dim=4):" << endl;
    token_emb.print();
    
    vector<int> sentence = {2, 5, 3, 8};  // A sentence of 4 tokens
    Tensor embedded = embedding(token_emb, sentence);
    cout << "Embedded sentence [2, 5, 3, 8]:" << endl;
    embedded.print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

// ===== EMBEDDING BASIC TEST =====
// Embedding weight (5 x 3):
// Tensor(5, 3):
// [0 , 1 , 2 ], [10 , 11 , 12 ], [20 , 21 , 22 ], [30 , 31 , 32 ], [40 , 41 , 42 ]
// embedding(indices=[0, 2, 4]):
// Tensor(3, 3):
// [0 , 1 , 2 ], [20 , 21 , 22 ], [40 , 41 , 42 ]

// ===== EMBEDDING WITH REPEATS =====
// embedding(indices=[1, 1, 3, 1]):
// Tensor(4, 3):
// [10 , 11 , 12 ], [10 , 11 , 12 ], [30 , 31 , 32 ], [10 , 11 , 12 ]

// ===== EMBEDDING SINGLE INDEX =====
// embedding(indices=[3]):
// Tensor(1, 3):
// [30 , 31 , 32 ]

// ===== EMBEDDING GRADIENT TEST =====
// Weight (4 x 2):
// Tensor(4, 2):
// [1 , 2 ], [3 , 4 ], [5 , 6 ], [7 , 8 ]
// embedding(indices=[0, 2, 3]):
// Tensor(3, 2):
// [1 , 2 ], [5 , 6 ], [7 , 8 ]
// Upstream gradient:
// Tensor(3, 2):
// [10 , 20 ], [30 , 40 ], [50 , 60 ]
// Weight gradient (row 1 should be 0):
// Tensor(4, 2):
// [10 , 20 ], [0 , 0 ], [30 , 40 ], [50 , 60 ]

// ===== EMBEDDING GRADIENT WITH REPEATS =====
// Weight (3 x 2):
// Tensor(3, 2):
// [1 , 2 ], [3 , 4 ], [5 , 6 ]
// embedding(indices=[0, 1, 0, 1, 0]):
// Tensor(5, 2):
// [1 , 2 ], [3 , 4 ], [1 , 2 ], [3 , 4 ], [1 , 2 ]
// Upstream gradient:
// Tensor(5, 2):
// [1 , 1 ], [2 , 2 ], [3 , 3 ], [4 , 4 ], [5 , 5 ]
// Weight gradient (row 0: 1+3+5=9, row 1: 2+4=6, row 2: 0):
// Tensor(3, 2):
// [9 , 9 ], [6 , 6 ], [0 , 0 ]

// ===== EMBEDDING FOR SEQUENCE =====
// Token embedding table (vocab=10, dim=4):
// Tensor(10, 4):
// [0 , 1 , 2 , 3 ], [10 , 11 , 12 , 13 ], [20 , 21 , 22 , 23 ], [30 , 31 , 32 , 33 ], [40 , 41 , 42 , 43 ], [50 , 51 , 52 , 53 ], [60 , 61 , 62 , 63 ], [70 , 71 , 72 , 73 ], [80 , 81 , 82 , 83 ], [90 , 91 , 92 , 93 ]
// Embedded sentence [2, 5, 3, 8]:
// Tensor(4, 4):
// [20 , 21 , 22 , 23 ], [50 , 51 , 52 , 53 ], [30 , 31 , 32 , 33 ], [80 , 81 , 82 , 83 ]

// ===== ALL TESTS COMPLETE =====
