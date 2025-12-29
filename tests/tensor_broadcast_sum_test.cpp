#include "tensor.h"
#include <iostream>

int main() {

    Tensor t1({2, 1, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);

    t1.print();
    cout<<"we got here"<<endl;
    Tensor t2 = t1.broadcast({2, 4, 3});
    t2.print();

    t2.set_grad(Tensor({2, 4, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false));
    t2.impl->backward_fn();

    t1.grad().print();

    Tensor t3 = t1.sum(1, true);
    t3.print();
    t3.set_grad(Tensor({2, 1, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false));
    t3.grad().print();
    t3.impl->backward_fn();
    t1.grad().print();

    Tensor t4({2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, true);
    t4.print();

    Tensor t5 = t4.sum(1, true);
    t5.print();
    t5.set_grad(Tensor({2, 1, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false));
    t5.impl->backward_fn();
    t4.grad().print();

    Tensor t6 = t4.sum(0, true);
    t6.print();
    t6.set_grad(Tensor({1, 2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, false));
    t6.impl->backward_fn();
    t4.grad().print();

    Tensor t7 = t4.sum(2, true);
    t7.print();
    t7.set_grad(Tensor({2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f}, false));
    t7.impl->backward_fn();
    t4.grad().print();

    Tensor t8 = t7.broadcast({2, 2, 3});
    t8.print();
    t8.set_grad(Tensor({2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, false));
    t8.impl->backward_fn();
    t7.grad().print();
}

// CORRECT OUTPUT

// Tensor(2, 1, 3):
// [[1 , 2 , 3 ]], [[4 , 5 , 6 ]]
// we got here
// Tensor(2, 4, 3):
// [[1 , 2 , 3 ], [1 , 2 , 3 ], [1 , 2 , 3 ], [1 , 2 , 3 ]], [[4 , 5 , 6 ], [4 , 5 , 6 ], [4 , 5 , 6 ], [4 , 5 , 6 ]]
// Tensor(2, 1, 3):
// [[4 , 4 , 4 ]], [[4 , 4 , 4 ]]
// Tensor(2, 1, 3):
// [[1 , 2 , 3 ]], [[4 , 5 , 6 ]]
// Tensor(2, 1, 3):
// [[1 , 1 , 1 ]], [[1 , 1 , 1 ]]
// Tensor(2, 1, 3):
// [[5 , 5 , 5 ]], [[5 , 5 , 5 ]]
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [4 , 5 , 6 ]], [[7 , 8 , 9 ], [10 , 11 , 12 ]]
// Tensor(2, 1, 3):
// [[5 , 7 , 9 ]], [[17 , 19 , 21 ]]
// Tensor(2, 2, 3):
// [[1 , 2 , 3 ], [1 , 2 , 3 ]], [[4 , 5 , 6 ], [4 , 5 , 6 ]]
// Tensor(1, 2, 3):
// [[8 , 10 , 12 ], [14 , 16 , 18 ]]
// Tensor(2, 2, 3):
// [[2 , 3 , 4 ], [2 , 3 , 4 ]], [[5 , 6 , 7 ], [5 , 6 , 7 ]]
// Tensor(2, 2, 1):
// [[6 ], [15 ]], [[24 ], [33 ]]
// Tensor(2, 2, 3):
// [[3 , 4 , 5 ], [4 , 5 , 6 ]], [[8 , 9 , 10 ], [9 , 10 , 11 ]]
// Tensor(2, 2, 3):
// [[6 , 6 , 6 ], [15 , 15 , 15 ]], [[24 , 24 , 24 ], [33 , 33 , 33 ]]
// Tensor(2, 2, 1):
// [[7 ], [17 ]], [[27 ], [37 ]]
