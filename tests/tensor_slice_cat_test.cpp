#include "tensor.h"
#include <iostream>
#include <cmath>

int main() {
    cout << "===== SLICE BASIC TEST =====" << endl;
    auto t1 = make_shared<Tensor>(vector<int>{3, 4}, vector<float>{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    }, true);
    cout << "Original:" << endl;
    t1->print();
    
    auto s1 = t1->slice(0, 0, 2);
    cout << "slice(0, 0, 2) - first 2 rows:" << endl;
    s1->print();
    
    auto s2 = t1->slice(1, 1, 3);
    cout << "slice(1, 1, 3) - columns 1-2:" << endl;
    s2->print();

    cout << "\n===== SLICE 3D TEST =====" << endl;
    auto t2 = make_shared<Tensor>(vector<int>{2, 3, 4}, vector<float>{
        1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12,
        13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24
    }, true);
    cout << "Original 2x3x4:" << endl;
    t2->print();
    
    auto s3 = t2->slice(1, 0, 2);
    cout << "slice(1, 0, 2):" << endl;
    s3->print();

    cout << "\n===== CAT BASIC TEST =====" << endl;
    auto a = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{7, 8, 9, 10, 11, 12}, true);
    cout << "Tensor A:" << endl;
    a->print();
    cout << "Tensor B:" << endl;
    b->print();
    
    auto c0 = cat({a, b}, 0);
    cout << "cat([A, B], axis=0):" << endl;
    c0->print();
    
    auto c1 = cat({a, b}, 1);
    cout << "cat([A, B], axis=1):" << endl;
    c1->print();

    cout << "\n===== CAT 3 TENSORS TEST =====" << endl;
    auto x = make_shared<Tensor>(vector<int>{1, 3}, vector<float>{1, 2, 3}, true);
    auto y = make_shared<Tensor>(vector<int>{1, 3}, vector<float>{4, 5, 6}, true);
    auto z = make_shared<Tensor>(vector<int>{1, 3}, vector<float>{7, 8, 9}, true);
    auto xyz = cat({x, y, z}, 0);
    cout << "cat([x, y, z], axis=0):" << endl;
    xyz->print();

    cout << "\n===== SLICE GRADIENT TEST =====" << endl;
    auto t3 = make_shared<Tensor>(vector<int>{3, 3}, vector<float>{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, true);
    cout << "Original:" << endl;
    t3->print();
    
    auto sliced = t3->slice(0, 1, 3);
    cout << "slice(0, 1, 3):" << endl;
    sliced->print();
    
    sliced->grad = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{10, 20, 30, 40, 50, 60}, false);
    sliced->backward();
    cout << "Gradient (should be 0 in first row, grad values in rows 1-2):" << endl;
    t3->grad->print();

    cout << "\n===== CAT GRADIENT TEST =====" << endl;
    auto p = make_shared<Tensor>(vector<int>{2, 2}, vector<float>{1, 2, 3, 4}, true);
    auto q = make_shared<Tensor>(vector<int>{2, 2}, vector<float>{5, 6, 7, 8}, true);
    cout << "P:" << endl;
    p->print();
    cout << "Q:" << endl;
    q->print();
    
    auto pq = cat({p, q}, 0);
    cout << "cat([P, Q], axis=0):" << endl;
    pq->print();
    
    pq->grad = make_shared<Tensor>(vector<int>{4, 2}, vector<float>{9, 10, 11, 12, 13, 14, 15, 16}, false);
    pq->backward();
    cout << "P gradient:" << endl;
    p->grad->print();
    cout << "Q gradient:" << endl;
    q->grad->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

