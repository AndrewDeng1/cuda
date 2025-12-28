#include "tensor.h"
#include <iostream>
#include <cassert>
#include <set>

int main() {
    cout << "===== MULTINOMIAL 1D WITH REPLACEMENT =====" << endl;
    auto probs = make_shared<Tensor>(vector<int>{4}, vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, false);
    auto samples = multinomial(probs, 10, true);
    assert(samples->shape[0] == 10);
    for(int i = 0; i < 10; i++) {
        assert(samples->at(i) >= 0 && samples->at(i) < 4);
    }
    cout << "PASSED" << endl;

    cout << "===== MULTINOMIAL 1D WITHOUT REPLACEMENT =====" << endl;
    auto samples2 = multinomial(probs, 4, false);
    assert(samples2->shape[0] == 4);
    std::set<int> unique;
    for(int i = 0; i < 4; i++) {
        unique.insert((int)samples2->at(i));
    }
    assert(unique.size() == 4);
    cout << "PASSED" << endl;

    cout << "===== MULTINOMIAL 2D BATCHED =====" << endl;
    auto probs2d = make_shared<Tensor>(vector<int>{3, 5}, vector<float>{
        0.1f, 0.2f, 0.3f, 0.2f, 0.2f,
        0.5f, 0.1f, 0.1f, 0.1f, 0.2f,
        0.0f, 0.0f, 0.5f, 0.5f, 0.0f
    }, false);
    auto samples3 = multinomial(probs2d, 3, true);
    assert(samples3->shape[0] == 3 && samples3->shape[1] == 3);
    for(int i = 0; i < samples3->size(); i++) {
        assert(samples3->at(i) >= 0 && samples3->at(i) < 5);
    }
    cout << "PASSED" << endl;

    cout << "===== MULTINOMIAL DETERMINISTIC =====" << endl;
    auto det_probs = make_shared<Tensor>(vector<int>{4}, vector<float>{0.0f, 0.0f, 1.0f, 0.0f}, false);
    auto det_samples = multinomial(det_probs, 5, true);
    for(int i = 0; i < 5; i++) {
        assert((int)det_samples->at(i) == 2);
    }
    cout << "PASSED" << endl;

    cout << "===== ALL MULTINOMIAL TESTS PASSED =====" << endl;
    return 0;
}

