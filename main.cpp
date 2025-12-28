#include "tensor.h"
#include <iostream>

int main() {
    cout << "===== SOFTMAX TEST =====" << endl;
    auto input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "Input:" << endl;
    input->print();

    auto out1 = softmax(input, 1);
    cout << "softmax(input, 1) - row-wise:" << endl;
    out1->print();
    // Expected: [0.0900, 0.2447, 0.6652] for each row

    auto out0 = softmax(input, 0);
    cout << "softmax(input, 0) - column-wise:" << endl;
    out0->print();
    // Expected: [0.0474, 0.9526] for each column

    cout << "\n===== SOFTMAX GRADIENT TEST =====" << endl;
    auto input2 = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );
    
    auto out2 = softmax(input2, 1);
    out2->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
        false
    );
    out2->backward();
    cout << "Gradient:" << endl;
    input2->grad->print();

    cout << "\n===== SOFTMAX LARGE VALUES (NUMERICAL STABILITY) =====" << endl;
    auto input3 = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{100.0f, 101.0f, 102.0f},
        true
    );
    cout << "Input:" << endl;
    input3->print();
    
    auto out3 = softmax(input3, 1);
    cout << "softmax - should be stable:" << endl;
    out3->print();
    // Expected: [0.0900, 0.2447, 0.6652]

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}
