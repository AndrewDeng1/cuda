#include "tensor.h"
#include <iostream>
#include <cmath>

int main() {
    cout << "===== SLICE BASIC TEST =====" << endl;
    Tensor t1({3, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    }, true);
    cout << "Original:" << endl;
    t1.print();
    
    Tensor s1 = t1.slice(0, 0, 2);
    cout << "slice(0, 0, 2) - first 2 rows:" << endl;
    s1.print();
    
    Tensor s2 = t1.slice(1, 1, 3);
    cout << "slice(1, 1, 3) - columns 1-2:" << endl;
    s2.print();

    cout << "\n===== SLICE 3D TEST =====" << endl;
    Tensor t2({2, 3, 4}, {
        1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12,
        13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24
    }, true);
    cout << "Original 2x3x4:" << endl;
    t2.print();
    
    Tensor s3 = t2.slice(1, 0, 2);
    cout << "slice(1, 0, 2):" << endl;
    s3.print();

    cout << "\n===== CAT BASIC TEST =====" << endl;
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    Tensor b({2, 3}, {7, 8, 9, 10, 11, 12}, true);
    cout << "Tensor A:" << endl;
    a.print();
    cout << "Tensor B:" << endl;
    b.print();
    
    Tensor c0 = cat({a, b}, 0);
    cout << "cat([A, B], axis=0):" << endl;
    c0.print();
    
    Tensor c1 = cat({a, b}, 1);
    cout << "cat([A, B], axis=1):" << endl;
    c1.print();

    cout << "\n===== CAT 3 TENSORS TEST =====" << endl;
    Tensor x({1, 3}, {1, 2, 3}, true);
    Tensor y({1, 3}, {4, 5, 6}, true);
    Tensor z({1, 3}, {7, 8, 9}, true);
    Tensor xyz = cat({x, y, z}, 0);
    cout << "cat([x, y, z], axis=0):" << endl;
    xyz.print();

    cout << "\n===== SLICE GRADIENT TEST =====" << endl;
    Tensor t3({3, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, true);
    cout << "Original:" << endl;
    t3.print();
    
    Tensor sliced = t3.slice(0, 1, 3);
    cout << "slice(0, 1, 3):" << endl;
    sliced.print();
    
    sliced.set_grad(Tensor({2, 3}, {10, 20, 30, 40, 50, 60}));
    sliced.backward();
    cout << "Gradient (should be 0 in first row, grad values in rows 1-2):" << endl;
    t3.grad().print();

    cout << "\n===== CAT GRADIENT TEST =====" << endl;
    Tensor p({2, 2}, {1, 2, 3, 4}, true);
    Tensor q({2, 2}, {5, 6, 7, 8}, true);
    cout << "P:" << endl;
    p.print();
    cout << "Q:" << endl;
    q.print();
    
    Tensor pq = cat({p, q}, 0);
    cout << "cat([P, Q], axis=0):" << endl;
    pq.print();
    
    pq.set_grad(Tensor({4, 2}, {9, 10, 11, 12, 13, 14, 15, 16}));
    pq.backward();
    cout << "P gradient:" << endl;
    p.grad().print();
    cout << "Q gradient:" << endl;
    q.grad().print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

// ===== SLICE BASIC TEST =====
// Original:
// Tensor(3, 4):
// [1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ], [9 , 10 , 11 , 12 ]
// slice(0, 0, 2) - first 2 rows:
// Tensor(2, 4):
// [1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ]
// slice(1, 1, 3) - columns 1-2:
// Tensor(3, 2):
// [2 , 3 ], [6 , 7 ], [10 , 11 ]

// ===== SLICE 3D TEST =====
// Original 2x3x4:
// Tensor(2, 3, 4):
// [[1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ], [9 , 10 , 11 , 12 ]], [[13 , 14 , 15 , 16 ], [17 , 18 , 19 , 20 ], [21 , 22 , 23 , 24 ]]
// slice(1, 0, 2):
// Tensor(2, 2, 4):
// [[1 , 2 , 3 , 4 ], [5 , 6 , 7 , 8 ]], [[13 , 14 , 15 , 16 ], [17 , 18 , 19 , 20 ]]

// ===== CAT BASIC TEST =====
// Tensor A:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// Tensor B:
// Tensor(2, 3):
// [7 , 8 , 9 ], [10 , 11 , 12 ]
// cat([A, B], axis=0):
// Tensor(4, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ], [10 , 11 , 12 ]
// cat([A, B], axis=1):
// Tensor(2, 6):
// [1 , 2 , 3 , 7 , 8 , 9 ], [4 , 5 , 6 , 10 , 11 , 12 ]

// ===== CAT 3 TENSORS TEST =====
// cat([x, y, z], axis=0):
// Tensor(3, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ]

// ===== SLICE GRADIENT TEST =====
// Original:
// Tensor(3, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ]
// slice(0, 1, 3):
// Tensor(2, 3):
// [4 , 5 , 6 ], [7 , 8 , 9 ]
// Gradient (should be 0 in first row, grad values in rows 1-2):
// Tensor(3, 3):
// [0 , 0 , 0 ], [10 , 20 , 30 ], [40 , 50 , 60 ]

// ===== CAT GRADIENT TEST =====
// P:
// Tensor(2, 2):
// [1 , 2 ], [3 , 4 ]
// Q:
// Tensor(2, 2):
// [5 , 6 ], [7 , 8 ]
// cat([P, Q], axis=0):
// Tensor(4, 2):
// [1 , 2 ], [3 , 4 ], [5 , 6 ], [7 , 8 ]
// P gradient:
// Tensor(2, 2):
// [9 , 10 ], [11 , 12 ]
// Q gradient:
// Tensor(2, 2):
// [13 , 14 ], [15 , 16 ]

// ===== ALL TESTS COMPLETE =====