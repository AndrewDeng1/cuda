#include "tensor.h"
#include <iostream>
#include <cmath>
#include <set>

int main() {
    cout << "===== ARANGE BASIC TEST =====" << endl;
    auto a1 = arange(0, 5, 1);
    cout << "arange(0, 5, 1):" << endl;
    a1->print();
    
    auto a2 = arange(1, 10, 2);
    cout << "arange(1, 10, 2):" << endl;
    a2->print();
    
    auto a3 = arange(0.0f, 1.0f, 0.25f);
    cout << "arange(0, 1, 0.25):" << endl;
    a3->print();

    cout << "\n===== ARANGE NEGATIVE STEP =====" << endl;
    auto a4 = arange(5, 0, -1);
    cout << "arange(5, 0, -1):" << endl;
    a4->print();

    cout << "\n===== MULTINOMIAL 1D =====" << endl;
    auto probs = make_shared<Tensor>(vector<int>{4}, vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, false);
    cout << "Probs: ";
    probs->print();
    
    cout << "5 samples with replacement:" << endl;
    for(int i = 0; i < 3; i++) {
        auto samples = multinomial(probs, 5, true);
        samples->print();
    }

    cout << "\n===== MULTINOMIAL WITHOUT REPLACEMENT =====" << endl;
    cout << "4 samples without replacement (should be all unique each time):" << endl;
    for(int i = 0; i < 3; i++) {
        auto samples = multinomial(probs, 4, false);
        samples->print();
        std::set<int> unique;
        for(int j = 0; j < 4; j++) {
            unique.insert((int)samples->at(j));
        }
        cout << "Unique count: " << unique.size() << endl;
    }

    cout << "\n===== MULTINOMIAL 2D BATCHED =====" << endl;
    auto probs2d = make_shared<Tensor>(vector<int>{2, 3}, vector<float>{
        0.5f, 0.3f, 0.2f,
        0.1f, 0.1f, 0.8f
    }, false);
    cout << "Probs:" << endl;
    probs2d->print();
    
    auto samples2d = multinomial(probs2d, 2, true);
    cout << "Samples shape: " << samples2d->shape[0] << "x" << samples2d->shape[1] << endl;
    samples2d->print();

    cout << "\n===== MULTINOMIAL DETERMINISTIC =====" << endl;
    auto det_probs = make_shared<Tensor>(vector<int>{4}, vector<float>{0.0f, 0.0f, 1.0f, 0.0f}, false);
    cout << "Probs [0, 0, 1, 0] - should always sample index 2:" << endl;
    auto det_samples = multinomial(det_probs, 5, true);
    det_samples->print();

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}
