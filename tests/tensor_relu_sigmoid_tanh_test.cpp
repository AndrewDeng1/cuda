#include "tensor.h"
#include <iostream>

int main() {
    // ===== ReLU TEST =====
    Tensor relu_input({2, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 3.0f}, true);

    Tensor relu_output = relu(relu_input);
    relu_output.print();  
    // Expected: [[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]]

    relu_output.set_grad(Tensor({2, 3}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, true));

    relu_output.backward();
    relu_input.grad().print();  
    // Expected: [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]

    // ===== SIGMOID TEST =====
    Tensor sigmoid_input({2, 2}, {0.0f, 1.0f, -1.0f, 2.0f}, true);

    Tensor sigmoid_output = sigmoid(sigmoid_input);
    sigmoid_output.print();  
    // ~[[0.5, 0.7311], [0.2689, 0.8808]]

    sigmoid_output.set_grad(Tensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}, true));

    sigmoid_output.backward();
    sigmoid_input.grad().print();  
    // ~[[0.25, 0.1966], [0.1966, 0.1049]]

    // ===== TANH TEST =====
    Tensor tanh_input({2, 2}, {-1.0f, 0.0f, 1.0f, 2.0f}, true);

    Tensor tanh_output = tanh_op(tanh_input);
    tanh_output.print();  
    // ~[[-0.7615, 0.0], [0.7615, 0.9640]]

    tanh_output.set_grad(Tensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}, true));

    tanh_output.backward();
    tanh_input.grad().print();  
    // ~[[0.4199, 1.0], [0.4199, 0.0707]]

    return 0;
}

// Tensor(2, 3):
// [0 , 0 , 1 ], [2 , 0 , 3 ]
// Tensor(2, 3):
// [0 , 0 , 1 ], [1 , 0 , 1 ]
// Tensor(2, 2):
// [0.5 , 0.731059 ], [0.268941 , 0.880797 ]
// Tensor(2, 2):
// [0.25 , 0.196612 ], [0.196612 , 0.104994 ]
// Tensor(2, 2):
// [-0.761594, 0 ], [0.761594 , 0.964028 ]
// Tensor(2, 2):
// [0.419974 , 1 ], [0.419974 , 0.0706509 ]
