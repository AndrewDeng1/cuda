#include "tensor.h"
#include <iostream>
#include <cmath>
#include <cfloat>

int main() {
    cout << "===== TRIL BASIC 2D TEST =====" << endl;
    auto t1 = make_shared<Tensor>(vector<int>{4, 4}, vector<float>{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }, true);
    cout << "Original:" << endl;
    t1->print();
    
    auto tl1 = tril(t1);
    cout << "tril (lower triangular, fill=0):" << endl;
    tl1->print();

    cout << "\n===== TRIL WITH DIAGONAL OFFSET =====" << endl;
    auto tl2 = tril(t1, 0.0f, 1);
    cout << "tril(diagonal=1) - include 1 above main diagonal:" << endl;
    tl2->print();
    
    auto tl3 = tril(t1, 0.0f, -1);
    cout << "tril(diagonal=-1) - exclude main diagonal:" << endl;
    tl3->print();

    cout << "\n===== TRIL FOR ATTENTION MASK (-inf) =====" << endl;
    auto t2 = make_shared<Tensor>(vector<int>{3, 3}, 1.0f, true);
    cout << "Original (all 1s):" << endl;
    t2->print();
    
    auto mask = tril(t2, -INFINITY);
    cout << "Attention mask (lower=1, upper=-inf):" << endl;
    mask->print();

    cout << "\n===== TRIL 3D (BATCHED) TEST =====" << endl;
    auto t3 = make_shared<Tensor>(vector<int>{2, 3, 3}, vector<float>{
        1, 2, 3,  4, 5, 6,  7, 8, 9,
        10, 11, 12,  13, 14, 15,  16, 17, 18
    }, true);
    cout << "Original 2x3x3:" << endl;
    t3->print();
    
    auto tl4 = tril(t3);
    cout << "tril (batched):" << endl;
    tl4->print();

    cout << "\n===== TRIL 4D (BATCH x HEADS) TEST =====" << endl;
    auto t4 = make_shared<Tensor>(vector<int>{2, 2, 3, 3}, vector<float>{
        1,2,3, 4,5,6, 7,8,9,
        1,2,3, 4,5,6, 7,8,9,
        1,2,3, 4,5,6, 7,8,9,
        1,2,3, 4,5,6, 7,8,9
    }, true);
    cout << "Original 2x2x3x3:" << endl;
    t4->print();
    
    auto tl5 = tril(t4);
    cout << "tril (4D batched):" << endl;
    tl5->print();

    cout << "\n===== TRIL GRADIENT TEST =====" << endl;
    auto t5 = make_shared<Tensor>(vector<int>{3, 3}, vector<float>{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, true);
    cout << "Original:" << endl;
    t5->print();
    
    auto tl6 = tril(t5);
    cout << "tril:" << endl;
    tl6->print();
    
    tl6->grad = make_shared<Tensor>(vector<int>{3, 3}, vector<float>{
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    }, false);
    tl6->backward();
    cout << "Gradient (should be 0 in upper triangle):" << endl;
    t5->grad->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

