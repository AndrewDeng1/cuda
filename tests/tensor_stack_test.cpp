#include "tensor.h"
#include <iostream>

int main() {
    cout << "===== STACK BASIC TEST (axis=0) =====" << endl;
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    Tensor b({2, 3}, {7, 8, 9, 10, 11, 12}, true);
    
    cout << "Tensor A (2x3):" << endl;
    a.print();
    cout << "Tensor B (2x3):" << endl;
    b.print();
    
    Tensor s0 = ::stack({a, b}, 0);
    cout << "stack([A, B], axis=0) -> (2, 2, 3):" << endl;
    s0.print();

    cout << "\n===== STACK AXIS=1 =====" << endl;
    Tensor s1 = ::stack({a, b}, 1);
    cout << "stack([A, B], axis=1) -> (2, 2, 3):" << endl;
    s1.print();

    cout << "\n===== STACK AXIS=2 (last) =====" << endl;
    Tensor s2 = ::stack({a, b}, 2);
    cout << "stack([A, B], axis=2) -> (2, 3, 2):" << endl;
    s2.print();

    cout << "\n===== STACK 3 TENSORS =====" << endl;
    Tensor c({2, 3}, {13, 14, 15, 16, 17, 18}, true);
    Tensor s3 = ::stack({a, b, c}, 0);
    cout << "stack([A, B, C], axis=0) -> (3, 2, 3):" << endl;
    s3.print();

    cout << "\n===== STACK 1D TENSORS =====" << endl;
    Tensor x({3}, {1, 2, 3}, true);
    Tensor y({3}, {4, 5, 6}, true);
    Tensor z({3}, {7, 8, 9}, true);
    
    cout << "x, y, z are 1D tensors of length 3" << endl;
    Tensor s4 = ::stack({x, y, z}, 0);
    cout << "stack([x, y, z], axis=0) -> (3, 3):" << endl;
    s4.print();
    
    Tensor s5 = ::stack({x, y, z}, 1);
    cout << "stack([x, y, z], axis=1) -> (3, 3):" << endl;
    s5.print();

    cout << "\n===== STACK GRADIENT TEST =====" << endl;
    Tensor p({2, 2}, {1, 2, 3, 4}, true);
    Tensor q({2, 2}, {5, 6, 7, 8}, true);
    
    cout << "P:" << endl;
    p.print();
    cout << "Q:" << endl;
    q.print();
    
    Tensor pq = ::stack({p, q}, 0);
    cout << "stack([P, Q], axis=0) -> (2, 2, 2):" << endl;
    pq.print();
    
    pq.set_grad(Tensor({2, 2, 2}, {
        10, 20, 30, 40,
        50, 60, 70, 80
    }));
    cout << "Upstream gradient:" << endl;
    pq.grad().print();
    
    pq.backward();
    cout << "P gradient:" << endl;
    p.grad().print();
    cout << "Q gradient:" << endl;
    q.grad().print();

    cout << "\n===== STACK NEGATIVE AXIS =====" << endl;
    Tensor s6 = ::stack({a, b}, -1);
    cout << "stack([A, B], axis=-1) -> (2, 3, 2):" << endl;
    s6.print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

// ===== STACK BASIC TEST (axis=0) =====
// Tensor A (2x3):
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// Tensor B (2x3):
// Tensor(2, 3):
// [7 , 8 , 9 ], [10 , 11 , 12 ]
// stack([A, B], axis=0) -> (2, 2, 3):
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [4 , 5 , 6 ]], [[7 , 8 , 9 ], [10 , 11 , 12 ]]

// ===== STACK AXIS=1 =====
// stack([A, B], axis=1) -> (2, 2, 3):
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [7 , 8 , 9 ]], [[4 , 5 , 6 ], [10 , 11 , 12 ]]

// ===== STACK AXIS=2 (last) =====
// stack([A, B], axis=2) -> (2, 3, 2):
// Tensor(2, 3, 2):
// [[1 , 7 ], [2 , 8 ], [3 , 9 ]], [[4 , 10 ], [5 , 11 ], [6 , 12 ]]

// ===== STACK 3 TENSORS =====
// stack([A, B, C], axis=0) -> (3, 2, 3):
// Tensor(3, 2, 3):
// [[1 , 2 , 3 ], [4 , 5 , 6 ]], [[7 , 8 , 9 ], [10 , 11 , 12 ]], [[13 , 14 , 15 ], [16 , 17 , 18 ]]

// ===== STACK 1D TENSORS =====
// x, y, z are 1D tensors of length 3
// stack([x, y, z], axis=0) -> (3, 3):
// Tensor(3, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ]
// stack([x, y, z], axis=1) -> (3, 3):
// Tensor(3, 3):
// [1 , 4 , 7 ], [2 , 5 , 8 ], [3 , 6 , 9 ]

// ===== STACK GRADIENT TEST =====
// P:
// Tensor(2, 2):
// [1 , 2 ], [3 , 4 ]
// Q:
// Tensor(2, 2):
// [5 , 6 ], [7 , 8 ]
// stack([P, Q], axis=0) -> (2, 2, 2):
// Tensor(2, 2, 2):
// [[1 , 2 ], [3 , 4 ]], [[5 , 6 ], [7 , 8 ]]
// Upstream gradient:
// Tensor(2, 2, 2):
// [[10 , 20 ], [30 , 40 ]], [[50 , 60 ], [70 , 80 ]]
// P gradient:
// Tensor(2, 2):
// [10 , 20 ], [30 , 40 ]
// Q gradient:
// Tensor(2, 2):
// [50 , 60 ], [70 , 80 ]

// ===== STACK NEGATIVE AXIS =====
// stack([A, B], axis=-1) -> (2, 3, 2):
// Tensor(2, 3, 2):
// [[1 , 7 ], [2 , 8 ], [3 , 9 ]], [[4 , 10 ], [5 , 11 ], [6 , 12 ]]

// ===== ALL TESTS COMPLETE =====