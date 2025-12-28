#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    cout << "===== ARANGE BASIC =====" << endl;
    auto a1 = arange(0, 5, 1);
    assert(a1->shape[0] == 5);
    assert(a1->at(0) == 0.0f);
    assert(a1->at(4) == 4.0f);
    cout << "PASSED" << endl;

    cout << "===== ARANGE STEP 2 =====" << endl;
    auto a2 = arange(1, 10, 2);
    assert(a2->shape[0] == 5);
    assert(a2->at(0) == 1.0f);
    assert(a2->at(4) == 9.0f);
    cout << "PASSED" << endl;

    cout << "===== ARANGE FLOAT STEP =====" << endl;
    auto a3 = arange(0.0f, 1.0f, 0.25f);
    assert(a3->shape[0] == 4);
    assert(abs(a3->at(0) - 0.0f) < 1e-5f);
    assert(abs(a3->at(3) - 0.75f) < 1e-5f);
    cout << "PASSED" << endl;

    cout << "===== ARANGE NEGATIVE STEP =====" << endl;
    auto a4 = arange(5, 0, -1);
    assert(a4->shape[0] == 5);
    assert(a4->at(0) == 5.0f);
    assert(a4->at(4) == 1.0f);
    cout << "PASSED" << endl;

    cout << "===== ARANGE EMPTY =====" << endl;
    auto a5 = arange(5, 0, 1);  // end < start with positive step
    assert(a5->shape[0] == 0);
    cout << "PASSED" << endl;

    cout << "===== ALL ARANGE TESTS PASSED =====" << endl;
    return 0;
}

