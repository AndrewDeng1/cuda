#include "tensor.h"
#include <iostream>

int main() {
    cout << "===== STACK BASIC TEST (axis=0) =====" << endl;
    auto a = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{7, 8, 9, 10, 11, 12}, true);
    
    cout << "Tensor A (2x3):" << endl;
    a->print();
    cout << "Tensor B (2x3):" << endl;
    b->print();
    
    auto s0 = stack({a, b}, 0);
    cout << "stack([A, B], axis=0) -> (2, 2, 3):" << endl;
    s0->print();

    cout << "\n===== STACK AXIS=1 =====" << endl;
    auto s1 = stack({a, b}, 1);
    cout << "stack([A, B], axis=1) -> (2, 2, 3):" << endl;
    s1->print();

    cout << "\n===== STACK AXIS=2 (last) =====" << endl;
    auto s2 = stack({a, b}, 2);
    cout << "stack([A, B], axis=2) -> (2, 3, 2):" << endl;
    s2->print();

    cout << "\n===== STACK 3 TENSORS =====" << endl;
    auto c = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{13, 14, 15, 16, 17, 18}, true);
    auto s3 = stack({a, b, c}, 0);
    cout << "stack([A, B, C], axis=0) -> (3, 2, 3):" << endl;
    s3->print();

    cout << "\n===== STACK 1D TENSORS =====" << endl;
    auto x = make_shared<Tensor>(vector<int>{3}, vector<float>{1, 2, 3}, true);
    auto y = make_shared<Tensor>(vector<int>{3}, vector<float>{4, 5, 6}, true);
    auto z = make_shared<Tensor>(vector<int>{3}, vector<float>{7, 8, 9}, true);
    
    cout << "x, y, z are 1D tensors of length 3" << endl;
    auto s4 = stack({x, y, z}, 0);
    cout << "stack([x, y, z], axis=0) -> (3, 3):" << endl;
    s4->print();
    
    auto s5 = stack({x, y, z}, 1);
    cout << "stack([x, y, z], axis=1) -> (3, 3):" << endl;
    s5->print();

    cout << "\n===== STACK GRADIENT TEST =====" << endl;
    auto p = make_shared<Tensor>(vector<int>{2, 2}, vector<float>{1, 2, 3, 4}, true);
    auto q = make_shared<Tensor>(vector<int>{2, 2}, vector<float>{5, 6, 7, 8}, true);
    
    cout << "P:" << endl;
    p->print();
    cout << "Q:" << endl;
    q->print();
    
    auto pq = stack({p, q}, 0);
    cout << "stack([P, Q], axis=0) -> (2, 2, 2):" << endl;
    pq->print();
    
    pq->grad = make_shared<Tensor>(vector<int>{2, 2, 2}, vector<float>{
        10, 20, 30, 40,
        50, 60, 70, 80
    }, false);
    cout << "Upstream gradient:" << endl;
    pq->grad->print();
    
    pq->backward();
    cout << "P gradient:" << endl;
    p->grad->print();
    cout << "Q gradient:" << endl;
    q->grad->print();

    cout << "\n===== STACK NEGATIVE AXIS =====" << endl;
    auto s6 = stack({a, b}, -1);
    cout << "stack([A, B], axis=-1) -> (2, 3, 2):" << endl;
    s6->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

