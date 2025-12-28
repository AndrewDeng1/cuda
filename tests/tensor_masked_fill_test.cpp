#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cfloat>

int main() {
    cout << "===== MASKED_FILL BASIC TEST =====" << endl;
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
    
    auto out = tensor->masked_fill(mask, -999.0f);
    assert(out->at({0, 0}) == 1.0f);
    assert(out->at({0, 1}) == -999.0f);
    assert(out->at({0, 2}) == -999.0f);
    assert(out->at({1, 2}) == -999.0f);
    assert(out->at({2, 2}) == 9.0f);
    cout << "PASSED" << endl;

    cout << "===== ATTENTION MASK PATTERN =====" << endl;
    auto scores = make_shared<Tensor>(vector<int>{4, 4}, vector<float>{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }, true);
    
    auto tril_mask = tril(4, 4);
    auto ones = make_shared<Tensor>(vector<int>{4, 4}, 1.0f, false);
    auto upper_mask = ones - tril_mask;
    
    auto masked = scores->masked_fill(upper_mask, -INFINITY);
    assert(masked->at({0, 0}) == 1.0f);
    assert(std::isinf(masked->at({0, 1})) && masked->at({0, 1}) < 0);
    assert(masked->at({1, 1}) == 6.0f);
    assert(std::isinf(masked->at({1, 2})) && masked->at({1, 2}) < 0);
    assert(masked->at({3, 3}) == 16.0f);
    cout << "PASSED" << endl;

    cout << "===== MASKED_FILL GRADIENT TEST =====" << endl;
    auto t = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        1, 2, 3,
        4, 5, 6
    }, true);
    auto m = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        0, 1, 0,
        1, 0, 0
    }, false);
    
    auto result = t->masked_fill(m, -100.0f);
    result->grad = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        10, 20, 30,
        40, 50, 60
    }, false);
    result->backward();
    
    assert(t->grad->at({0, 0}) == 10.0f);
    assert(t->grad->at({0, 1}) == 0.0f);  // masked
    assert(t->grad->at({0, 2}) == 30.0f);
    assert(t->grad->at({1, 0}) == 0.0f);  // masked
    assert(t->grad->at({1, 1}) == 50.0f);
    assert(t->grad->at({1, 2}) == 60.0f);
    cout << "PASSED" << endl;

    cout << "===== ALL MASKED_FILL TESTS PASSED =====" << endl;
    return 0;
}
