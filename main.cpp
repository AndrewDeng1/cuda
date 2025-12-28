#include "tensor.h"
#include <iostream>

int main() {
    cout << "===== RANDINT BASIC =====" << endl;
    auto r1 = randint(0, 10, {5});
    cout << "randint(0, 10, {5}):" << endl;
    r1->print();
    
    cout << "\n===== RANDINT 2D =====" << endl;
    auto r2 = randint(0, 100, {3, 4});
    cout << "randint(0, 100, {3, 4}):" << endl;
    r2->print();
    
    cout << "\n===== RANDINT RANGE CHECK =====" << endl;
    auto r3 = randint(5, 8, {20});
    cout << "randint(5, 8, {20}) - should be 5, 6, or 7:" << endl;
    r3->print();
    
    bool in_range = true;
    for(int i = 0; i < r3->size(); i++) {
        int v = (int)r3->at(i);
        if(v < 5 || v >= 8) in_range = false;
    }
    cout << "All in range [5, 8): " << (in_range ? "YES" : "NO") << endl;

    cout << "\n===== ALL TESTS COMPLETE =====" << endl;
    return 0;
}
