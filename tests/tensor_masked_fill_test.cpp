#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cfloat>

int main() {
    cout << "===== MASKED_FILL BASIC TEST =====" << endl;
    Tensor tensor({3, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, true);
    Tensor mask({3, 3}, {
        0, 1, 1,
        0, 0, 1,
        0, 0, 0
    }, false);
    
    Tensor out = tensor.masked_fill(mask, -999.0f);
    assert(out.at({0, 0}) == 1.0f);
    assert(out.at({0, 1}) == -999.0f);
    assert(out.at({0, 2}) == -999.0f);
    assert(out.at({1, 2}) == -999.0f);
    assert(out.at({2, 2}) == 9.0f);
    cout << "PASSED" << endl;

    cout << "===== ATTENTION MASK PATTERN =====" << endl;
    Tensor scores({4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    }, true);
    
    Tensor tril_mask = tril(4, 4);
    Tensor ones({4, 4}, 1.0f, false);
    Tensor upper_mask = ones - tril_mask;
    
    Tensor masked = scores.masked_fill(upper_mask, -INFINITY);
    assert(masked.at({0, 0}) == 1.0f);
    assert(std::isinf(masked.at({0, 1})) && masked.at({0, 1}) < 0);
    assert(masked.at({1, 1}) == 6.0f);
    assert(std::isinf(masked.at({1, 2})) && masked.at({1, 2}) < 0);
    assert(masked.at({3, 3}) == 16.0f);
    cout << "PASSED" << endl;

    cout << "===== MASKED_FILL GRADIENT TEST =====" << endl;
    Tensor t({2, 3}, {
        1, 2, 3,
        4, 5, 6
    }, true);
    Tensor m({2, 3}, {
        0, 1, 0,
        1, 0, 0
    }, false);
    
    Tensor result = t.masked_fill(m, -100.0f);
    result.set_grad(Tensor({2, 3}, {
        10, 20, 30,
        40, 50, 60
    }, false));
    result.backward();
    
    assert(t.grad().at({0, 0}) == 10.0f);
    assert(t.grad().at({0, 1}) == 0.0f);  // masked
    assert(t.grad().at({0, 2}) == 30.0f);
    assert(t.grad().at({1, 0}) == 0.0f);  // masked
    assert(t.grad().at({1, 1}) == 50.0f);
    assert(t.grad().at({1, 2}) == 60.0f);
    cout << "PASSED" << endl;

    cout << "===== ALL MASKED_FILL TESTS PASSED =====" << endl;
    return 0;
}

// ===== MASKED_FILL BASIC TEST =====
// PASSED
// ===== ATTENTION MASK PATTERN =====
// PASSED
// ===== MASKED_FILL GRADIENT TEST =====
// PASSED
// ===== ALL MASKED_FILL TESTS PASSED =====
