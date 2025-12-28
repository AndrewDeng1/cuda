#include "tensor.h"
#include <iostream>
#include <cmath>

int main() {
    cout << "===== DROPOUT BASIC TEST (p=0.5, training=true) =====" << endl;
    auto t1 = make_shared<Tensor>(vector<int>{2, 4}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, true);
    cout << "Input:" << endl;
    t1->print();
    
    auto d1 = dropout(t1, 0.5f, true);
    cout << "After dropout (p=0.5, training=true):" << endl;
    d1->print();
    
    int zeros = 0;
    for(int i = 0; i < d1->size(); i++) {
        if(d1->at(i) == 0.0f) zeros++;
    }
    cout << "Zeros: " << zeros << " out of " << d1->size() << endl;

    cout << "\n===== DROPOUT INFERENCE TEST (training=false) =====" << endl;
    auto t2 = make_shared<Tensor>(vector<int>{2, 4}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, true);
    cout << "Input:" << endl;
    t2->print();
    
    auto d2 = dropout(t2, 0.5f, false);
    cout << "After dropout (p=0.5, training=false) - should be unchanged:" << endl;
    d2->print();
    
    bool unchanged = true;
    for(int i = 0; i < t2->size(); i++) {
        if(abs(t2->at(i) - d2->at(i)) > 1e-5f) unchanged = false;
    }
    cout << "Output unchanged: " << (unchanged ? "YES" : "NO") << endl;

    cout << "\n===== DROPOUT SCALING TEST =====" << endl;
    auto t3 = make_shared<Tensor>(vector<int>{1, 4}, vector<float>{2.0f, 2.0f, 2.0f, 2.0f}, true);
    cout << "Input (all 2.0):" << endl;
    t3->print();
    
    auto d3 = dropout(t3, 0.5f, true);
    cout << "After dropout (p=0.5) - non-zero values should be 4.0 (scaled by 1/(1-0.5)=2):" << endl;
    d3->print();
    
    bool correct_scale = true;
    for(int i = 0; i < d3->size(); i++) {
        if(d3->at(i) != 0.0f && abs(d3->at(i) - 4.0f) > 1e-5f) correct_scale = false;
    }
    cout << "Scaling correct: " << (correct_scale ? "YES" : "NO") << endl;

    cout << "\n===== DROPOUT GRADIENT TEST =====" << endl;
    auto t4 = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);
    cout << "Input:" << endl;
    t4->print();
    
    auto d4 = dropout(t4, 0.3f, true);
    cout << "After dropout (p=0.3):" << endl;
    d4->print();
    
    d4->grad = make_shared<Tensor>(vector<int>{2, 3}, 1.0f, false);
    d4->backward();
    
    cout << "Input gradient (should match dropout pattern):" << endl;
    t4->grad->print();
    
    bool grad_matches = true;
    float scale = 1.0f / (1.0f - 0.3f);
    for(int i = 0; i < d4->size(); i++) {
        if(d4->at(i) == 0.0f && t4->grad->at(i) != 0.0f) grad_matches = false;
        if(d4->at(i) != 0.0f && abs(t4->grad->at(i) - scale) > 1e-5f) grad_matches = false;
    }
    cout << "Gradient pattern matches: " << (grad_matches ? "YES" : "NO") << endl;

    cout << "\n===== DROPOUT p=0 TEST =====" << endl;
    auto t5 = make_shared<Tensor>(vector<int>{2, 2}, vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, true);
    auto d5 = dropout(t5, 0.0f, true);
    cout << "Input:" << endl;
    t5->print();
    cout << "After dropout (p=0) - should be unchanged:" << endl;
    d5->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}

