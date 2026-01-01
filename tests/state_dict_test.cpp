#include "../nn.h"
#include "../tensor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

// Hyperparameters matching gpt.cpp
const int n_embd = 128;
const int n_head = 2;
const int n_layer = 2;
const int vocab_size = 65;  // Typical vocab size for character-level
const int block_size = 64;
const float dropout_p = 0.2f;

// Helper function to convert tensor to JSON representation
string tensor_to_json(const Tensor& t) {
    ostringstream oss;
    vector<int> shape = t.shape();
    int total_size = t.size();
    
    oss << "{";
    oss << "\"shape\":[";
    for (size_t i = 0; i < shape.size(); i++) {
        oss << shape[i];
        if (i < shape.size() - 1) oss << ",";
    }
    oss << "],";
    oss << "\"data\":[";
    for (int i = 0; i < total_size; i++) {
        oss << fixed << setprecision(8) << t.at(i);
        if (i < total_size - 1) oss << ",";
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

// Helper function to save state dict to JSON file
void save_state_dict_to_json(const map<string, Tensor>& state_dict, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }
    
    file << "{\n";
    size_t idx = 0;
    for (const auto& [key, tensor] : state_dict) {
        file << "  \"" << key << "\": " << tensor_to_json(tensor);
        if (idx < state_dict.size() - 1) {
            file << ",";
        }
        file << "\n";
        idx++;
    }
    file << "}\n";
    file.close();
    cout << "State dict saved to " << filename << endl;
}

// Helper function to load state dict from JSON (simplified - just for testing)
// In a real implementation, you'd use a JSON parser
map<string, Tensor> load_state_dict_from_json(const string& filename) {
    // For this test, we'll just verify the file exists and has content
    // A full implementation would parse JSON and reconstruct tensors
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }
    
    // Read file content
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    cout << "Loaded JSON file with " << content.size() << " bytes" << endl;
    
    // Return empty map for now - full JSON parsing would go here
    // This is just to demonstrate the structure
    return map<string, Tensor>();
}

int main() {
    cout << "=== Testing state_dict() and load_state_dict() ===" << endl;
    
    try {
        // Create a GPT model (simplified version that matches gpt.cpp structure)
        cout << "\n1. Creating GPT model..." << endl;
        
        // Create a simple module that contains the GPT architecture
        class SimpleGPT : public Module {
        public:
            Embedding token_embedding_table_;
            Embedding position_embedding_table_;
            Sequential blocks_;
            LayerNorm ln_f_;
            Linear lm_head_;
            
            SimpleGPT() 
                : token_embedding_table_(Embedding(vocab_size, n_embd)),
                  position_embedding_table_(Embedding(block_size, n_embd)),
                  ln_f_(LayerNorm(n_embd)),
                  lm_head_(Linear(n_embd, vocab_size)) {
                // Add simple blocks (using Linear for simplicity)
                for (int i = 0; i < n_layer; i++) {
                    blocks_.append(Linear(n_embd, n_embd));
                }
            }
            
            void register_submodules() override {
                register_module("token_embedding_table", &token_embedding_table_);
                register_module("position_embedding_table", &position_embedding_table_);
                register_module("blocks", &blocks_);
                register_module("ln_f", &ln_f_);
                register_module("lm_head", &lm_head_);
            }
            
            Tensor forward(const Tensor& x) override {
                // Simple forward pass
                Tensor tok_emb = token_embedding_table_.forward(x);
                Tensor out = blocks_.forward(tok_emb);
                out = ln_f_.forward(out);
                out = lm_head_.forward(out);
                return out;
            }
        };
        
        SimpleGPT model;
        
        cout << "Model created successfully" << endl;
        
        // Get state dict
        cout << "\n2. Getting state_dict()..." << endl;
        map<string, Tensor> state_dict = model.state_dict();
        
        cout << "State dict contains " << state_dict.size() << " parameters/buffers:" << endl;
        for (const auto& [key, tensor] : state_dict) {
            vector<int> shape = tensor.shape();
            cout << "  " << key << ": shape [";
            for (size_t i = 0; i < shape.size(); i++) {
                cout << shape[i];
                if (i < shape.size() - 1) cout << ", ";
            }
            cout << "], size " << tensor.size() << endl;
        }
        
        // Save to JSON
        cout << "\n3. Saving state_dict to JSON..." << endl;
        save_state_dict_to_json(state_dict, "sample_gpt_state_dict.json");
        
        // Save to safetensors
        cout << "\n3b. Saving state_dict to safetensors..." << endl;
        write_safe_tensors(state_dict, "sample_gpt_state_dict.safetensors");
        
        // Test load_state_dict by creating a new model and loading
        cout << "\n4. Testing load_state_dict()..." << endl;
        SimpleGPT model2;
        
        // Get initial parameter values
        map<string, Tensor> state_dict2_before = model2.state_dict();
        cout << "Before loading, first parameter value: " << state_dict2_before.begin()->second.at(0) << endl;
        
        // Load the state dict
        auto [missing_keys, unexpected_keys] = model2.load_state_dict(state_dict, true);
        
        cout << "Missing keys: " << missing_keys.size() << endl;
        if (!missing_keys.empty()) {
            for (const auto& key : missing_keys) {
                cout << "  - " << key << endl;
            }
        }
        
        cout << "Unexpected keys: " << unexpected_keys.size() << endl;
        if (!unexpected_keys.empty()) {
            for (const auto& key : unexpected_keys) {
                cout << "  - " << key << endl;
            }
        }
        
        // Verify the loaded state
        map<string, Tensor> state_dict2_after = model2.state_dict();
        cout << "\n5. Verifying loaded state..." << endl;
        
        bool all_match = true;
        for (const auto& [key, original_tensor] : state_dict) {
            auto it = state_dict2_after.find(key);
            if (it == state_dict2_after.end()) {
                cout << "ERROR: Key " << key << " not found in loaded model!" << endl;
                all_match = false;
                continue;
            }
            
            Tensor& loaded_tensor = it->second;
            if (original_tensor.shape() != loaded_tensor.shape()) {
                cout << "ERROR: Shape mismatch for " << key << endl;
                all_match = false;
                continue;
            }
            
            // Check first few values
            bool values_match = true;
            int check_count = min(5, (int)original_tensor.size());
            for (int i = 0; i < check_count; i++) {
                if (abs(original_tensor.at(i) - loaded_tensor.at(i)) > 1e-6f) {
                    values_match = false;
                    break;
                }
            }
            
            if (!values_match) {
                cout << "ERROR: Values don't match for " << key << endl;
                cout << "  Original[0] = " << original_tensor.at(0) << ", Loaded[0] = " << loaded_tensor.at(0) << endl;
                all_match = false;
            } else {
                cout << "  " << key << ": OK (shape matches, values match)" << endl;
            }
        }
        
        if (all_match) {
            cout << "\n✓ SUCCESS: All parameters loaded correctly!" << endl;
        } else {
            cout << "\n✗ FAILED: Some parameters didn't load correctly" << endl;
            return 1;
        }
        
        cout << "\n=== Test completed ===" << endl;
        cout << "Check gpt_state_dict.json to see the saved state dict" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}

