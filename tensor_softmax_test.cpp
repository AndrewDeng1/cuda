#include "tensor.h"
#include <iostream>

int main() {
    // ===== SOFTMAX FUNCTION TEST =====
    auto softmax_input = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        true
    );

    cout << "SOFTMAX TEST - Input tensor:" << endl;
    softmax_input->print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    auto softmax_output_axis1 = softmax_input->softmax(1, true);
    cout << "SOFTMAX along axis 1 (softmax of each row):" << endl;
    softmax_output_axis1->print();  
    // Expected: Row 1: exp(1)=2.718, exp(2)=7.389, exp(3)=20.086, sum=30.193
    // Expected: Row 1: [0.0900, 0.2447, 0.6652]
    // Expected: Row 2: exp(4)=54.598, exp(5)=148.413, exp(6)=403.429, sum=606.440
    // Expected: Row 2: [0.0900, 0.2447, 0.6652]
    // Expected: Tensor(2, 3): [0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]

    softmax_output_axis1->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
        true
    );

    softmax_output_axis1->backward();
    cout << "Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    auto softmax_output_axis0 = softmax_input->softmax(0, true);
    cout << "SOFTMAX along axis 0 (softmax of each column):" << endl;
    softmax_output_axis0->print();  
    // Expected: Col 1: exp(1)=2.718, exp(4)=54.598, sum=57.316
    // Expected: Col 1: [0.0474, 0.9526]
    // Expected: Col 2: exp(2)=7.389, exp(5)=148.413, sum=155.802
    // Expected: Col 2: [0.0474, 0.9526]
    // Expected: Col 3: exp(3)=20.086, exp(6)=403.429, sum=423.515
    // Expected: Col 3: [0.0474, 0.9526]
    // Expected: Tensor(2, 3): [0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]

    softmax_output_axis0->grad = make_shared<Tensor>(
        vector<int>{2, 3},
        vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    softmax_output_axis0->backward();
    cout << "Gradients for SOFTMAX along axis 0:" << endl;
    softmax_input->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    // Test with negative values to see softmax behavior
    auto softmax_input2 = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{-2.0f, 2.0f, -1.0f, 1.0f},
        true
    );

    cout << "\nSecond SOFTMAX test - Input tensor with negative values:" << endl;
    softmax_input2->print();
    // Expected: Tensor(2, 2): [-2, 2], [-1, 1]

    auto softmax_output2 = softmax_input2->softmax(1, true);
    cout << "SOFTMAX along axis 1 (with negative values):" << endl;
    softmax_output2->print();  
    // Expected: Row 1: exp(-2)=0.1353, exp(2)=7.3891, sum=7.5244
    // Expected: Row 1: [0.0180, 0.9820]
    // Expected: Row 2: exp(-1)=0.3679, exp(1)=2.7183, sum=3.0862
    // Expected: Row 2: [0.1192, 0.8808]
    // Expected: Tensor(2, 2): [0.0180, 0.9820], [0.1192, 0.8808]

    softmax_output2->grad = make_shared<Tensor>(
        vector<int>{2, 2},
        vector<float>{1.0f, 0.0f, 0.0f, 1.0f},
        true
    );

    softmax_output2->backward();
    cout << "Gradients for SOFTMAX with negative values:" << endl;
    softmax_input2->grad->print();  
    // Expected: Gradients showing how changes in softmax output affect input

    // Test with large values to check numerical stability
    auto softmax_input3 = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{100.0f, 101.0f, 102.0f},
        true
    );

    cout << "\nThird SOFTMAX test - Input tensor with large values:" << endl;
    softmax_input3->print();
    // Expected: Tensor(1, 3): [100, 101, 102]

    auto softmax_output3 = softmax_input3->softmax(1, true);
    cout << "SOFTMAX with large values (should be numerically stable):" << endl;
    softmax_output3->print();  
    // Expected: Should handle large values without overflow
    // Expected: exp(100), exp(101), exp(102) are very large but ratios should be correct
    // Expected: Tensor(1, 3): [0.0900, 0.2447, 0.6652] (approximate ratios)

    softmax_output3->grad = make_shared<Tensor>(
        vector<int>{1, 3},
        vector<float>{1.0f, 0.0f, 0.0f},
        true
    );

    softmax_output3->backward();
    cout << "Gradients for SOFTMAX with large values:" << endl;
    softmax_input3->grad->print();  
    // Expected: Gradients should be computed correctly even with large input values

    return 0;
}



// SOFTMAX TEST - Input tensor:
// Tensor(2, 3):
// [1 , 2 , 3 ], [4 , 5 , 6 ]
// SOFTMAX along axis 1 (softmax of each row):
// Tensor(2, 3):
// [0.0900306 , 0.244728 , 0.665241 ], [0.0900306 , 0.244728 , 0.665241 ]
// Gradients for SOFTMAX along axis 1:
// Tensor(2, 3):
// [0.0819251 , -0.022033, -0.059892], [-0.022033, 0.184836 , -0.162803]
// SOFTMAX along axis 0 (softmax of each column):
// Tensor(2, 3):
// [0.0474259 , 0.0474259 , 0.0474259 ], [0.952574 , 0.952574 , 0.952574 ]
// Gradients for SOFTMAX along axis 0:
// Tensor(2, 3):
// [0.127102 , -0.022033, -0.105069], [-0.0672097, 0.184836 , -0.117627]

// Second SOFTMAX test - Input tensor with negative values:
// Tensor(2, 2):
// [-2, 2 ], [-1, 1 ]
// SOFTMAX along axis 1 (with negative values):
// Tensor(2, 2):
// [0.0179862 , 0.982014 ], [0.119203 , 0.880797 ]
// Gradients for SOFTMAX with negative values:
// Tensor(2, 2):
// [0.0176627 , -0.0176627], [-0.104994, 0.104994 ]

// Third SOFTMAX test - Input tensor with large values:
// Tensor(1, 3):
// [100 , 101 , 102 ]
// SOFTMAX with large values (should be numerically stable):
// Tensor(1, 3):
// [0.0900306 , 0.244728 , 0.665241 ]
// Gradients for SOFTMAX with large values:
// Tensor(1, 3):
// [0.0819251 , -0.022033, -0.059892]