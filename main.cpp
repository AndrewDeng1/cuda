#include "tensor.h"
#include <iostream>
#include <cmath>
#include <cfloat>

int main() {
    cout << "===== TRIL BASIC TEST =====" << endl;
    auto t1 = tril(4, 4);
    cout << "tril(4, 4) - lower triangular 1s, upper 0s:" << endl;
    t1->print();

    cout << "\n===== TRIL NON-SQUARE =====" << endl;
    auto t2 = tril(3, 5);
    cout << "tril(3, 5):" << endl;
    t2->print();
    
    auto t3 = tril(5, 3);
    cout << "tril(5, 3):" << endl;
    t3->print();

    cout << "\n===== MASKED_FILL BASIC TEST =====" << endl;
    auto tensor = make_shared<Tensor>(vector<int>{3, 3}, vector<float>{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, true);
    auto mask = make_shared<Tensor>(vector<int>{3, 3}, vector<float>{
        0, 1, 1,
        0, 0, 1,
        0, 0, 0
    }, false);
    
    cout << "Input:" << endl;
    tensor->print();
    cout << "Mask (1=fill, 0=keep):" << endl;
    mask->print();
    
    auto out1 = tensor->masked_fill(mask, -999.0f);
    cout << "masked_fill(mask, -999):" << endl;
    out1->print();

    cout << "\n===== ATTENTION MASK PATTERN =====" << endl;
    auto scores = make_shared<Tensor>(vector<int>{4, 4}, vector<float>{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }, true);
    cout << "Attention scores:" << endl;
    scores->print();
    
    // Create upper triangular mask (inverse of tril)
    auto tril_mask = tril(4, 4);
    auto ones = make_shared<Tensor>(vector<int>{4, 4}, 1.0f, false);
    auto upper_mask = ones - tril_mask;  // 1s in upper triangle
    cout << "Upper triangle mask (1 - tril):" << endl;
    upper_mask->print();
    
    auto masked_scores = scores->masked_fill(upper_mask, -INFINITY);
    cout << "Masked scores (upper=-inf):" << endl;
    masked_scores->print();

    cout << "\n===== MASKED_FILL GRADIENT TEST =====" << endl;
    auto t4 = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        1, 2, 3,
        4, 5, 6
    }, true);
    auto m4 = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        0, 1, 0,
        1, 0, 0
    }, false);
    
    cout << "Input:" << endl;
    t4->print();
    
    auto out4 = t4->masked_fill(m4, -100.0f);
    cout << "masked_fill:" << endl;
    out4->print();
    
    out4->grad = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        10, 20, 30,
        40, 50, 60
    }, false);
    out4->backward();
    cout << "Gradient (0 where masked):" << endl;
    t4->grad->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}
