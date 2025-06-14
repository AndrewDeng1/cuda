#include "tensor.h"
#include <iostream>

int main() {

    // // Create a 2x3 tensor
    vector<int> shape = {2, 4, 3};
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    auto t1 = make_shared<Tensor>(shape, data, true);
    
    vector<float>alt_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
    auto t2 = make_shared<Tensor>(shape, alt_data, true);
    auto t3 = t1*t2;
    t3->print();
    t3->grad = make_shared<Tensor>(shape, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}, true);
    t3->grad->print();
    t3->backward();
    t1->grad->print();
    t2->grad->print();

    auto t4 = make_shared<Tensor>(vector<int>{3, 1}, vector<float>{1.0f, 2.0f, 3.0f}, true);
    auto t5 = make_shared<Tensor>(vector<int>{2, 1, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    auto t6 = t4*t5;
    t6->print();
    t6->grad = make_shared<Tensor>(vector<int>{2, 3, 3}, vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0}, true);
    // t6->grad = make_shared<Tensor>(vector<int>{2, 3, 3}, vector<float>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, true);
    t6->backward();
    t4->grad->print();
    t5->grad->print();
}

// Tensor(2, 4, 3):
// [[1 , 4 , 9 ], [16 , 25 , 36 ], [49 , 64 , 81 ], [100 , 121 , 144 ]], [[169 , 196 , 225 ], [256 , 289 , 324 ], [361 , 400 , 441 ], [484 , 529 , 576 ]]
// Tensor(2, 4, 3):
// [[1 , 2 , 3 ], [4 , 5 , 6 ], [7 , 8 , 9 ], [10 , 11 , 12 ]], [[13 , 14 , 15 ], [16 , 17 , 18 ], [19 , 20 , 21 ], [22 , 23 , 24 ]]
// q size: 1
// Backward *
// Backward broadcast
// Backward broadcast
// Tensor(2, 4, 3):
// [[1 , 4 , 9 ], [16 , 25 , 36 ], [49 , 64 , 81 ], [100 , 121 , 144 ]], [[169 , 196 , 225 ], [256 , 289 , 324 ], [361 , 400 , 441 ], [484 , 529 , 576 ]]
// Tensor(2, 4, 3):
// [[1 , 4 , 9 ], [16 , 25 , 36 ], [49 , 64 , 81 ], [100 , 121 , 144 ]], [[169 , 196 , 225 ], [256 , 289 , 324 ], [361 , 400 , 441 ], [484 , 529 , 576 ]]
// Tensor(2, 3, 3):
// [[1 , 2 , 3 ], [2 , 4 , 6 ], [3 , 6 , 9 ]], [[4 , 5 , 6 ], [8 , 10 , 12 ], [12 , 15 , 18 ]]        
// q size: 1
// Backward *
// Backward broadcast
// Backward broadcast
// Tensor(3, 1):
// [181 ], [244 ], [307 ]
// Tensor(2, 1, 3):
// [[30 , 36 , 42 ]], [[84 , 90 , 96 ]]