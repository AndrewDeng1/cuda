#include "tensor.h"
#include <iostream>

int main() {
    // ===== SOFTMAX FUNCTION TEST =====
    Tensor softmax_input({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);

    cout << "SOFTMAX TEST - Input tensor:" << endl;
    softmax_input.print();
    // Expected: Tensor(2, 3): [1, 2, 3], [4, 5, 6]

    Tensor softmax_output_axis1 = softmax(softmax_input, 1);
    cout << "SOFTMAX along axis 1 (softmax of each row):" << endl;
    softmax_output_axis1.print();  
    // Expected: Tensor(2, 3): [0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]

    softmax_output_axis1.set_grad(Tensor({2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}));

    softmax_output_axis1.backward();
    cout << "Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input.grad().print();  

    Tensor softmax_output_axis0 = softmax(softmax_input, 0);
    cout << "SOFTMAX along axis 0 (softmax of each column):" << endl;
    softmax_output_axis0.print();  
    // Expected: Tensor(2, 3): [0.0474, 0.0474, 0.0474], [0.9526, 0.9526, 0.9526]

    softmax_output_axis0.set_grad(Tensor({2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}));

    softmax_output_axis0.backward();
    cout << "Gradients for SOFTMAX along axis 0:" << endl;
    softmax_input.grad().print();  

    // Test with negative values
    Tensor softmax_input2({2, 2}, {-2.0f, 2.0f, -1.0f, 1.0f}, true);

    cout << "\nSecond SOFTMAX test - Input tensor with negative values:" << endl;
    softmax_input2.print();
    // Expected: Tensor(2, 2): [-2, 2], [-1, 1]

    Tensor softmax_output2 = softmax(softmax_input2, 1);
    cout << "SOFTMAX along axis 1 (with negative values):" << endl;
    softmax_output2.print();  
    // Expected: Tensor(2, 2): [0.0180, 0.9820], [0.1192, 0.8808]

    softmax_output2.set_grad(Tensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}));

    softmax_output2.backward();
    cout << "Gradients for SOFTMAX with negative values:" << endl;
    softmax_input2.grad().print();  

    // Test with large values for numerical stability
    Tensor softmax_input3({1, 3}, {100.0f, 101.0f, 102.0f}, true);

    cout << "\nThird SOFTMAX test - Input tensor with large values:" << endl;
    softmax_input3.print();
    // Expected: Tensor(1, 3): [100, 101, 102]

    Tensor softmax_output3 = softmax(softmax_input3, 1);
    cout << "SOFTMAX with large values (should be numerically stable):" << endl;
    softmax_output3.print();  
    // Expected: Tensor(1, 3): [0.0900, 0.2447, 0.6652]

    softmax_output3.set_grad(Tensor({1, 3}, {1.0f, 0.0f, 0.0f}));

    softmax_output3.backward();
    cout << "Gradients for SOFTMAX with large values:" << endl;
    softmax_input3.grad().print();  

    // ===== HIGHER DIMENSION SOFTMAX TEST =====
    cout << "\n" << string(50, '=') << endl;
    cout << "HIGHER DIMENSION SOFTMAX TEST" << endl;
    cout << string(50, '=') << endl;

    // Create 3D input tensor (batch_size=2, sequence_length=3, num_classes=4)
    Tensor softmax_input_3d({2, 3, 4}, {
        // Batch 0
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        // Batch 1
        1.3f, 1.4f, 1.5f, 1.6f,
        1.7f, 1.8f, 1.9f, 2.0f,
        2.1f, 2.2f, 2.3f, 2.4f
    }, true);

    cout << "3D Input tensor shape: [" << softmax_input_3d.shape()[0] << ", " 
         << softmax_input_3d.shape()[1] << ", " << softmax_input_3d.shape()[2] << "]" << endl;
    cout << "3D Input tensor:" << endl;
    softmax_input_3d.print();

    // Test softmax along the last axis (axis 2) - class probabilities
    Tensor softmax_output_3d_axis2 = softmax(softmax_input_3d, 2);
    cout << "3D SOFTMAX along axis 2 (class probabilities):" << endl;
    softmax_output_3d_axis2.print();

    softmax_output_3d_axis2.set_grad(Tensor({2, 3, 4}, {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    }));

    softmax_output_3d_axis2.backward();
    cout << "3D Gradients for SOFTMAX along axis 2:" << endl;
    softmax_input_3d.grad().print();

    // Test softmax along the sequence axis (axis 1) - attention weights
    Tensor softmax_output_3d_axis1 = softmax(softmax_input_3d, 1);
    cout << "3D SOFTMAX along axis 1 (attention weights):" << endl;
    softmax_output_3d_axis1.print();

    softmax_output_3d_axis1.set_grad(Tensor({2, 3, 4}, {
        1.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    }));

    softmax_output_3d_axis1.backward();
    cout << "3D Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input_3d.grad().print();

    // ===== 4D TENSOR TEST (BATCH, CHANNELS, HEIGHT, WIDTH) =====
    cout << "\n" << string(50, '=') << endl;
    cout << "4D TENSOR SOFTMAX TEST" << endl;
    cout << string(50, '=') << endl;

    // Create 4D input tensor (batch_size=2, channels=3, height=2, width=2)
    Tensor softmax_input_4d({2, 3, 2, 2}, {
        // Batch 0, Channel 0-2
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        // Batch 1, Channel 0-2
        1.3f, 1.4f, 1.5f, 1.6f,
        1.7f, 1.8f, 1.9f, 2.0f,
        2.1f, 2.2f, 2.3f, 2.4f
    }, true);

    cout << "4D Input tensor shape: [" << softmax_input_4d.shape()[0] << ", " 
         << softmax_input_4d.shape()[1] << ", " << softmax_input_4d.shape()[2] << ", " 
         << softmax_input_4d.shape()[3] << "]" << endl;
    cout << "4D Input tensor:" << endl;
    softmax_input_4d.print();

    // Test softmax along the channel axis (axis 1)
    Tensor softmax_output_4d_axis1 = softmax(softmax_input_4d, 1);
    cout << "4D SOFTMAX along axis 1 (channel probabilities):" << endl;
    softmax_output_4d_axis1.print();

    softmax_output_4d_axis1.set_grad(Tensor({2, 3, 2, 2}, {
        // Batch 0
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        // Batch 1
        0.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    }));

    softmax_output_4d_axis1.backward();
    cout << "4D Gradients for SOFTMAX along axis 1:" << endl;
    softmax_input_4d.grad().print();

    // Test softmax along axis 2
    Tensor softmax_output_4d_axis2 = softmax(softmax_input_4d, 2);
    cout << "4D SOFTMAX along axis 2 (height attention):" << endl;
    softmax_output_4d_axis2.print();

    softmax_output_4d_axis2.set_grad(Tensor({2, 3, 2, 2}, {
        // Batch 0
        1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f,
        // Batch 1
        1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 0.0f
    }));

    softmax_output_4d_axis2.backward();
    cout << "4D Gradients for SOFTMAX along axis 2:" << endl;
    softmax_input_4d.grad().print();

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

// ==================================================
// HIGHER DIMENSION SOFTMAX TEST
// ==================================================
// 3D Input tensor shape: [2, 3, 4]
// 3D Input tensor:
// Tensor(2, 3, 4):
// [[0.1 , 0.2 , 0.3 , 0.4 ], [0.5 , 0.6 , 0.7 , 0.8 ], [0.9 , 1 , 1.1 , 1.2 ]], [[1.3 , 1.4 , 1.5 , 1.6 ], [1.7 , 1.8 , 1.9 , 2 ], [2.1 , 2.2 , 2.3 , 2.4 ]]
// 3D SOFTMAX along axis 2 (class probabilities):
// Tensor(2, 3, 4):
// [[0.213838 , 0.236328 , 0.261183 , 0.288651 ], [0.213838 , 0.236328 , 0.261183 , 0.288651 ], [0.213838 , 0.236328 , 0.261183 , 0.288651 ]], [[0.213838 , 0.236328 , 0.261183 , 0.288651 ], [0.213838 , 0.236328 , 0.261183 , 0.288651 ], [0.213838 , 0.236328 , 0.261183 , 0.288651 ]]
// 3D Gradients for SOFTMAX along axis 2:
// Tensor(2, 3, 4):
// [[0.168111 , -0.0505359, -0.0558508, -0.0617247], [-0.0505359, 0.180477 , -0.0617247, -0.0682163], [-0.0558508, -0.0617247, 0.192966 , -0.0753907]], [[-0.0617247, -0.0682163, -0.0753907, 0.205332 ], [0.168111 , -0.0505359, -0.0558508, -0.0617247], [-0.0505359, 0.180477 , -0.0617247, -0.0682163]]
// 3D SOFTMAX along axis 1 (attention weights):
// Tensor(2, 3, 4):
// [[0.211983 , 0.211983 , 0.211983 , 0.211983 ], [0.316241 , 0.316241 , 0.316241 , 0.316241 ], [0.471776 , 0.471776 , 0.471776 , 0.471776 ]], [[0.211983 , 0.211983 , 0.211983 , 0.211983 ], [0.316241 , 0.316241 , 0.316241 , 0.316241 ], [0.471776 , 0.471776 , 0.471776 , 0.471776 ]]
// 3D Gradients for SOFTMAX along axis 1:
// Tensor(2, 3, 4):
// [[0.335158 , 0.11651 , 0.111195 , 0.105321 ], [-0.117574, 0.113439 , -0.128762, -0.135254], [-0.155859, -0.161733, 0.0929578 , -0.175399]], [[-0.128762, -0.135254, -0.142428, 0.138294 ], [0.384344 , 0.165697 , 0.160382 , 0.154508 ], [-0.199731, 0.0312819 , -0.21092, -0.217411]]

// ==================================================
// 4D TENSOR SOFTMAX TEST
// ==================================================
// 4D Input tensor shape: [2, 3, 2, 2]
// 4D Input tensor:
// Tensor(2, 3, 2, 2):
// [[[0.1 , 0.2 ], [0.3 , 0.4 ]], [[0.5 , 0.6 ], [0.7 , 0.8 ]], [[0.9 , 1 ], [1.1 , 1.2 ]]], [[[1.3 , 1.4 ], [1.5 , 1.6 ]], [[1.7 , 1.8 ], [1.9 , 2 ]], [[2.1 , 2.2 ], [2.3 , 2.4 ]]]
// 4D SOFTMAX along axis 1 (channel probabilities):
// Tensor(2, 3, 2, 2):
// [[[0.211983 , 0.211983 ], [0.211983 , 0.211983 ]], [[0.316241 , 0.316241 ], [0.316241 , 0.316241 ]], [[0.471776 , 0.471776 ], [0.471776 , 0.471776 ]]], [[[0.211983 , 0.211983 ], [0.211983 , 0.211983 ]], [[0.316241 , 0.316241 ], [0.316241 , 0.316241 ]], [[0.471776 , 0.471776 ], [0.471776 , 0.471776 ]]]
// 4D Gradients for SOFTMAX along axis 1:
// Tensor(2, 3, 2, 2):
// [[[0.167046 , -0.0670376], [-0.100008, 0.167046 ]], [[-0.0670376, 0.216233 ], [-0.149195, -0.0670376]], [[-0.100008, -0.149195], [0.249203 , -0.100008]]], [[[-0.0670376, 0.167046 ], [-0.100008, -0.0670376]], [[0.216233 , -0.0670376], [-0.149195, 0.216233 ]], [[-0.149195, -0.100008], [0.249203 , -0.149195]]]
// 4D SOFTMAX along axis 2 (height attention):
// Tensor(2, 3, 2, 2):
// [[[0.450166 , 0.450166 ], [0.549834 , 0.549834 ]], [[0.450166 , 0.450166 ], [0.549834 , 0.549834 ]], [[0.450166 , 0.450166 ], [0.549834 , 0.549834 ]]], [[[0.450166 , 0.450166 ], [0.549834 , 0.549834 ]], [[0.450166 , 0.450166 ], [0.549834 , 0.549834 ]], [[0.450166 , 0.450166 ], [0.549834 , 0.549834 ]]]
// 4D Gradients for SOFTMAX along axis 2:
// Tensor(2, 3, 2, 2):
// [[[0.414563 , 0.180479 ], [-0.347525, -0.0804705]], [[0.180479 , 0.463749 ], [-0.396712, -0.314554]], [[0.147508 , 0.0983216 ], [0.00168683 , -0.347525]]], [[[0.180479 , 0.414563 ], [-0.347525, -0.314554]], [[0.463749 , 0.180479 ], [-0.396712, -0.0312839]], [[0.0983216 , 0.147508 ], [0.00168683 , -0.396712]]]
