#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    cout << "===== TRIL BASIC TEST =====" << endl;
    auto t1 = tril(4, 4);
    assert(t1->at({0, 0}) == 1.0f);
    assert(t1->at({0, 1}) == 0.0f);
    assert(t1->at({1, 0}) == 1.0f);
    assert(t1->at({1, 1}) == 1.0f);
    assert(t1->at({3, 3}) == 1.0f);
    assert(t1->at({0, 3}) == 0.0f);
    cout << "PASSED" << endl;

    cout << "===== TRIL NON-SQUARE =====" << endl;
    auto t2 = tril(3, 5);
    assert(t2->shape[0] == 3 && t2->shape[1] == 5);
    assert(t2->at({0, 0}) == 1.0f);
    assert(t2->at({0, 1}) == 0.0f);
    assert(t2->at({2, 2}) == 1.0f);
    assert(t2->at({2, 4}) == 0.0f);
    
    auto t3 = tril(5, 3);
    assert(t3->shape[0] == 5 && t3->shape[1] == 3);
    assert(t3->at({4, 2}) == 1.0f);
    assert(t3->at({0, 2}) == 0.0f);
    cout << "PASSED" << endl;

    cout << "===== TRIL 1x1 =====" << endl;
    auto t4 = tril(1, 1);
    assert(t4->at(0) == 1.0f);
    cout << "PASSED" << endl;

    cout << "===== ALL TRIL TESTS PASSED =====" << endl;
    return 0;
}

