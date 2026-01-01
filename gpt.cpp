#include "nn.h"
#include "optim.h"
#include "tensor.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>
#include <cfloat>

using namespace std;

// Hyperparameters
const int batch_size = 32;
const int block_size = 64;
const int max_iters = 40;
// const int max_iters = 3000;
// const int eval_interval = 300;
const int eval_interval = 100;
const int save_iters = 5;  // Save checkpoint every N iterations
const float learning_rate = 1e-3f;
// const int eval_iters = 50;
const int eval_iters = 1;
const int n_embd = 128;
const int n_head = 2;
const int n_layer = 2;
// const int n_head = 4;
// const int n_layer = 3;
const float dropout_p = 0.2f;

const string checkpoint_file = "checkpoint_iter_30.safetensors";
const bool save_checkpoint = true;
const bool load_checkpoint = true;

// Global variables
int vocab_size;
unordered_map<char, int> char_to_idx;
unordered_map<int, char> idx_to_char;
vector<int> train_data;
vector<int> val_data;

// Global random number generator with fixed seed (like torch.manual_seed(67))
mt19937 rng(67);

// Forward declarations
string decode(const vector<int>& indices);
vector<int> encode(const string& s);
pair<Tensor, Tensor> get_batch(const string& split);
map<string, float> estimate_loss(Module& model);

// ============================================================================
// Head: one head of self-attention
// ============================================================================
class Head : public Module {
public:
    Linear key_;
    Linear query_;
    Linear value_;
    Tensor tril_mask_;
    Dropout dropout_;

    Head(int head_size) 
        : key_(Linear(n_embd, head_size, false)),
          query_(Linear(n_embd, head_size, false)),
          value_(Linear(n_embd, head_size, false)),
          tril_mask_(tril(block_size, block_size)),
          dropout_(Dropout(dropout_p)) {
        // No need to call register - lazy registration handles it
    }

    void register_buffers() override {
        register_buffer("tril", &tril_mask_);
    }

    void register_submodules() override {
        register_module("key", &key_);
        register_module("query", &query_);
        register_module("value", &value_);
        register_module("dropout", &dropout_);
    }

    Tensor forward(const Tensor& x) override {
        // x: (B, T, C)
        vector<int> x_shape = x.shape();
        int B = x_shape[0];
        int T = x_shape[1];
        int C = x_shape[2];
        // cout << "[Head::forward] Input shape: [" << B << ", " << T << ", " << C << "]" << endl;

        // Generate K and Q matrices
        Tensor k = key_.forward(x);  // (B, T, head_size)
        // cout << "[Head::forward] k shape: [";
        // for(int i = 0; i < k.shape().size(); i++) {
        //     cout << k.shape()[i];
        //     if(i < k.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        
        Tensor q = query_.forward(x);  // (B, T, head_size)
        // cout << "[Head::forward] q shape: [";
        // for(int i = 0; i < q.shape().size(); i++) {
        //     cout << q.shape()[i];
        //     if(i < q.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        // Compute attention scores: (Q @ K^T) / sqrt(head_size)
        // cout << "[Head::forward] Computing kT = k.transpose(-2, -1)" << endl;
        Tensor kT = k.transpose(-2, -1);  // (B, head_size, T)
        // cout << "[Head::forward] kT shape: [";
        // for(int i = 0; i < kT.shape().size(); i++) {
        //     cout << kT.shape()[i];
        //     if(i < kT.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        
        // cout << "[Head::forward] Computing wei = matmul(q, kT)" << endl;
        Tensor wei = matmul(q, kT);  // Matrix multiplication: (B, T, T)
        // cout << "[Head::forward] wei shape: [";
        // for(int i = 0; i < wei.shape().size(); i++) {
        //     cout << wei.shape()[i];
        //     if(i < wei.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        
        int head_size = k.shape().back();
        float scale = 1.0f / sqrtf(head_size);
        wei = wei * scale;

        // Mask future tokens - equivalent to: wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        // tril[:T, :T] == 0 means upper triangle (where tril is 0)
        // cout << "[Head::forward] Creating mask" << endl;
        Tensor tril_slice = tril_mask_.slice(0, 0, T).slice(1, 0, T);  // (T, T)
        Tensor ones_mask = ones({T, T});
        Tensor upper_mask = ones_mask - tril_slice;  // Upper triangle is 1, lower is 0
        
        // Broadcast mask from (T, T) to (B, T, T) for masked_fill
        // Reshape to (1, T, T) then broadcast
        // cout << "[Head::forward] Broadcasting mask from (T, T) to (B, T, T)" << endl;
        upper_mask = upper_mask.reshape({1, T, T});
        upper_mask = upper_mask.broadcast({B, T, T}, false);
        // cout << "[Head::forward] upper_mask shape after broadcast: [";
        // for(int i = 0; i < upper_mask.shape().size(); i++) {
        //     cout << upper_mask.shape()[i];
        //     if(i < upper_mask.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        
        // masked_fill where mask != 0 (i.e., upper triangle) with -inf
        // cout << "[Head::forward] Applying masked_fill" << endl;
        wei = wei.masked_fill(upper_mask, -FLT_MAX);
        // cout << "[Head::forward] masked_fill complete" << endl;

        // Softmax
        // cout << "[Head::forward] Computing softmax" << endl;
        wei = softmax(wei, -1);  // (B, T, T)

        // Dropout
        // cout << "[Head::forward] Applying dropout" << endl;
        wei = dropout_.forward(wei);

        // Weighted aggregation
        // cout << "[Head::forward] Computing value projection" << endl;
        Tensor v = value_.forward(x);  // (B, T, head_size)
        // cout << "[Head::forward] v shape: [";
        // for(int i = 0; i < v.shape().size(); i++) {
        //     cout << v.shape()[i];
        //     if(i < v.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        
        // cout << "[Head::forward] Computing matmul(wei, v)" << endl;
        Tensor out = matmul(wei, v);  // (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        // cout << "[Head::forward] out shape: [";
        // for(int i = 0; i < out.shape().size(); i++) {
        //     cout << out.shape()[i];
        //     if(i < out.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        return out;
    }
};

// ============================================================================
// MultiHeadAttention: multiple heads of self-attention in parallel
// ============================================================================
class MultiHeadAttention : public Module {
public:
    ModuleList heads_;
    Linear proj_;
    Dropout dropout_;

    MultiHeadAttention(int num_heads, int head_size)
        : proj_(Linear(head_size * num_heads, n_embd)),
          dropout_(Dropout(dropout_p)) {
        // Create heads
        for(int i = 0; i < num_heads; i++) {
            heads_.append(Head(head_size));
        }
    }

    void register_submodules() override {
        register_module("heads", &heads_);
        register_module("proj", &proj_);
        register_module("dropout", &dropout_);
    }

    Tensor forward(const Tensor& x) override {
        // Pass through each head and concatenate
        vector<Tensor> head_outputs;
        for(size_t i = 0; i < heads_.size(); i++) {
            head_outputs.push_back(heads_[i]->forward(x));
        }
        
        // Concatenate along last dimension
        // cout << "[MultiHeadAttention] Concatenating " << head_outputs.size() << " heads" << endl;
        Tensor out = cat(head_outputs, -1);  // (B, T, head_size * num_heads)
        // cout << "[MultiHeadAttention] After cat, shape: [";
        // for(int i = 0; i < out.shape().size(); i++) {
        //     cout << out.shape()[i];
        //     if(i < out.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        // Project and dropout
        // cout << "[MultiHeadAttention] Applying projection" << endl;
        out = proj_.forward(out);
        // cout << "[MultiHeadAttention] After projection, shape: [";
        // for(int i = 0; i < out.shape().size(); i++) {
        //     cout << out.shape()[i];
        //     if(i < out.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        // cout << "[MultiHeadAttention] Applying dropout" << endl;
        out = dropout_.forward(out);

        return out;
    }
};

// ============================================================================
// FeedForward: simple linear layer followed by non-linearity
// ============================================================================
class FeedForward : public Module {
public:
    Sequential net_;

    FeedForward(int n_embd) {
        net_.append(Linear(n_embd, 4 * n_embd));
        net_.append(ReLU());
        net_.append(Linear(4 * n_embd, n_embd));
        net_.append(Dropout(dropout_p));
    }

    void register_submodules() override {
        register_module("net", &net_);
    }

    Tensor forward(const Tensor& x) override {
        return net_.forward(x);
    }
};

// ============================================================================
// Block: Transformer block
// ============================================================================
class Block : public Module {
public:
    MultiHeadAttention sa_;
    FeedForward ffwd_;
    LayerNorm ln1_;
    LayerNorm ln2_;

    Block(int n_embd, int n_head) 
        : sa_(MultiHeadAttention(n_head, n_embd / n_head)),
          ffwd_(FeedForward(n_embd)),
          ln1_(LayerNorm(n_embd)),
          ln2_(LayerNorm(n_embd)) {
    }

    void register_submodules() override {
        register_module("sa", &sa_);
        register_module("ffwd", &ffwd_);
        register_module("ln1", &ln1_);
        register_module("ln2", &ln2_);
    }

    Tensor forward(const Tensor& x) override {
        // Pre-norm: x = x + sa(ln1(x))
        // cout << "[Block] Applying ln1" << endl;
        Tensor x1 = ln1_.forward(x);
        // cout << "[Block] Applying multi-head attention" << endl;
        Tensor x2 = sa_.forward(x1);
        // cout << "[Block] Adding residual connection" << endl;
        Tensor x3 = x + x2;

        // Pre-norm: x = x + ffwd(ln2(x))
        // cout << "[Block] Applying ln2" << endl;
        Tensor x4 = ln2_.forward(x3);
        // cout << "[Block] Applying feedforward" << endl;
        Tensor x5 = ffwd_.forward(x4);
        // cout << "[Block] Adding final residual connection" << endl;
        Tensor out = x3 + x5;

        return out;
    }
};

// ============================================================================
// GPTLanguageModel
// ============================================================================
class GPTLanguageModel : public Module {
public:
    Embedding token_embedding_table_;
    Embedding position_embedding_table_;
    Sequential blocks_;
    LayerNorm ln_f_;
    Linear lm_head_;

    GPTLanguageModel()
        : token_embedding_table_(Embedding(vocab_size, n_embd)),
          position_embedding_table_(Embedding(block_size, n_embd)),
          ln_f_(LayerNorm(n_embd)),
          lm_head_(Linear(n_embd, vocab_size)) {
        
        // Create blocks
        for(int i = 0; i < n_layer; i++) {
            blocks_.append(Block(n_embd, n_head));
        }

        // Initialize weights (simplified - just use default initialization)
        // In PyTorch: normal_(mean=0.0, std=0.02) for Linear and Embedding
    }

    void register_submodules() override {
        register_module("token_embedding_table", &token_embedding_table_);
        register_module("position_embedding_table", &position_embedding_table_);
        register_module("blocks", &blocks_);
        register_module("ln_f", &ln_f_);
        register_module("lm_head", &lm_head_);
    }

    // Required by Module base class
    Tensor forward(const Tensor& x) override {
        auto [logits, _] = forward_with_targets(x, nullptr);
        return logits;
    }

    pair<Tensor, Tensor> forward_with_targets(const Tensor& idx, const Tensor* targets = nullptr) {
        // static int forward_count = 0;
        // forward_count++;
        // cout << "\n[GPT::forward] === Forward pass #" << forward_count << " ===" << endl;
        // idx: (B, T)
        vector<int> idx_shape = idx.shape();
        int B = idx_shape[0];
        int T = idx_shape[1];
        // cout << "[GPT::forward] Input idx shape: [" << B << ", " << T << "]" << endl;

        // Token embeddings
        // Convert idx to proper format for embedding
        // cout << "[GPT::forward] Computing token embeddings" << endl;
        Tensor tok_emb = token_embedding_table_.forward(idx);  // (B, T, C)
        // cout << "[GPT::forward] tok_emb shape: [";
        // for(int i = 0; i < tok_emb.shape().size(); i++) {
        //     cout << tok_emb.shape()[i];
        //     if(i < tok_emb.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        // Position embeddings - equivalent to torch.arange(T)
        // cout << "[GPT::forward] Computing position embeddings" << endl;
        Tensor pos_indices_tensor = arange(0.0f, (float)T, 1.0f);
        Tensor pos_emb = position_embedding_table_.forward(pos_indices_tensor);  // (T, C)
        // cout << "[GPT::forward] pos_emb shape (before broadcast): [";
        // for(int i = 0; i < pos_emb.shape().size(); i++) {
        //     cout << pos_emb.shape()[i];
        //     if(i < pos_emb.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;
        
        // Broadcast pos_emb from (T, C) to (B, T, C) for addition
        // Reshape to (1, T, C) then broadcast
        // cout << "[GPT::forward] Broadcasting pos_emb to (B, T, C)" << endl;
        pos_emb = pos_emb.reshape({1, T, n_embd});
        pos_emb = pos_emb.broadcast({B, T, n_embd}, false);
        // cout << "[GPT::forward] pos_emb shape (after broadcast): [";
        // for(int i = 0; i < pos_emb.shape().size(); i++) {
        //     cout << pos_emb.shape()[i];
        //     if(i < pos_emb.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        // Add embeddings
        // cout << "[GPT::forward] Adding token and position embeddings" << endl;
        Tensor x = tok_emb + pos_emb;  // (B, T, C)
        // cout << "[GPT::forward] x shape after embedding addition: [";
        // for(int i = 0; i < x.shape().size(); i++) {
        //     cout << x.shape()[i];
        //     if(i < x.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        // Pass through blocks
        // cout << "[GPT::forward] Passing through transformer blocks" << endl;
        x = blocks_.forward(x);  // (B, T, C)
        // cout << "[GPT::forward] x shape after blocks: [";
        // for(int i = 0; i < x.shape().size(); i++) {
        //     cout << x.shape()[i];
        //     if(i < x.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        // Final layer norm
        // cout << "[GPT::forward] Applying final layer norm" << endl;
        x = ln_f_.forward(x);  // (B, T, C)

        // Language model head
        // cout << "[GPT::forward] Computing logits" << endl;
        Tensor logits = lm_head_.forward(x);  // (B, T, vocab_size)
        // cout << "[GPT::forward] logits shape: [";
        // for(int i = 0; i < logits.shape().size(); i++) {
        //     cout << logits.shape()[i];
        //     if(i < logits.shape().size() - 1) cout << ", ";
        // }
        // cout << "]" << endl;

        Tensor loss;
        if(targets != nullptr) {
            // cout << "[GPT::forward] Computing loss" << endl;
            // cout << "[GPT::forward] targets shape: [";
            // for(int i = 0; i < targets->shape().size(); i++) {
            //     cout << targets->shape()[i];
            //     if(i < targets->shape().size() - 1) cout << ", ";
            // }
            // cout << "]" << endl;
            
            // Reshape logits and targets
            // cout << "[GPT::forward] Reshaping logits to (B*T, vocab_size)" << endl;
            Tensor logits_flat = logits.reshape({B * T, vocab_size});
            // cout << "[GPT::forward] Reshaping targets to (B*T,)" << endl;
            Tensor targets_flat = targets->reshape({B * T});

            // Cross entropy loss
            // cout << "[GPT::forward] Computing cross_entropy" << endl;
            loss = cross_entropy(logits_flat, targets_flat);
            // cout << "[GPT::forward] Loss computed: " << loss.at(0) << endl;
        }

        return make_pair(logits, loss);
    }

    Tensor generate(const Tensor& idx, int max_new_tokens) {
        // idx: (B, T)
        Tensor context = idx;
        
        for(int i = 0; i < max_new_tokens; i++) {
            // Crop to last block_size tokens
            int T_curr = context.shape()[1];
            int start = max(0, T_curr - block_size);
            Tensor idx_cond = context.slice(1, start, T_curr);

            // Get predictions
            auto [logits, _] = forward_with_targets(idx_cond, nullptr);
            
            // Focus on last time step
            Tensor logits_last = logits.slice(1, logits.shape()[1] - 1, logits.shape()[1]);  // (B, 1, vocab_size)
            logits_last = logits_last.reshape({logits_last.shape()[0], vocab_size});  // (B, vocab_size)

            // Softmax to get probabilities
            Tensor probs = softmax(logits_last, -1);  // (B, vocab_size)

            // Sample from distribution
            Tensor idx_next = multinomial(probs, 1, false);  // (B, 1) - contains sampled indices as floats
            
            // Convert idx_next to proper shape for concatenation
            // idx_next is (B, 1), need to reshape and convert to int indices
            vector<float> idx_next_vals;
            for(int i = 0; i < idx_next.size(); i++) {
                idx_next_vals.push_back(idx_next.at(i));
            }
            
            // Reshape to (B, 1) for concatenation
            Tensor idx_next_reshaped({(int)idx_next_vals.size(), 1}, idx_next_vals, false);
            
            // Concatenate along dimension 1: (B, T) + (B, 1) -> (B, T+1)
            vector<Tensor> to_cat = {context, idx_next_reshaped};
            context = cat(to_cat, 1);  // (B, T+1)
        }

        return context;
    }
};

// ============================================================================
// Utility functions
// ============================================================================

string decode(const vector<int>& indices) {
    string result;
    for(int idx : indices) {
        if(idx_to_char.find(idx) != idx_to_char.end()) {
            result += idx_to_char[idx];
        }
    }
    return result;
}

vector<int> encode(const string& s) {
    vector<int> result;
    for(char c : s) {
        if(char_to_idx.find(c) != char_to_idx.end()) {
            result.push_back(char_to_idx[c]);
        }
    }
    return result;
}

pair<Tensor, Tensor> get_batch(const string& split) {
    const vector<int>& data = (split == "train") ? train_data : val_data;
    
    // Generate random indices using randint
    Tensor start_indices = randint(0, data.size() - block_size, {batch_size});
    
    vector<Tensor> x_batches;
    vector<Tensor> y_batches;
    
    for(int i = 0; i < batch_size; i++) {
        int start_idx = (int)start_indices.at(i);
        
        // Get x: data[start_idx:start_idx+block_size]
        vector<float> x_vals;
        for(int j = start_idx; j < start_idx + block_size; j++) {
            x_vals.push_back((float)data[j]);
        }
        x_batches.push_back(Tensor({block_size}, x_vals, false));
        
        // Get y: data[start_idx+1:start_idx+block_size+1]
        vector<float> y_vals;
        for(int j = start_idx + 1; j < start_idx + block_size + 1; j++) {
            y_vals.push_back((float)data[j]);
        }
        y_batches.push_back(Tensor({block_size}, y_vals, false));
    }
    
    // Stack to get (B, T)
    Tensor x = ::stack(x_batches, 0);  // (B, T) - use global stack function
    Tensor y = ::stack(y_batches, 0);  // (B, T)
    
    return make_pair(x, y);
}

map<string, float> estimate_loss(Module& model) {
    model.eval();
    
    map<string, float> out;
    
    for(const string& split : {"train", "val"}) {
        vector<float> losses;
        
        for(int k = 0; k < eval_iters; k++) {
            // cout << "[estimate_loss] " << split << " split, iteration " << k << " of " << eval_iters << endl;
            auto [X, Y] = get_batch(split);
            
            GPTLanguageModel* gpt_model = dynamic_cast<GPTLanguageModel*>(&model);
            if(gpt_model) {
                auto [logits, loss] = gpt_model->forward_with_targets(X, &Y);
                losses.push_back(loss.at(0));
            }
        }
        
        // Compute mean
        float mean_loss = 0.0f;
        for(float l : losses) {
            mean_loss += l;
        }
        mean_loss /= losses.size();
        
        out[split] = mean_loss;
    }
    
    model.train();
    return out;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    // Random seed is set via global rng(67) above

    // Read input file
    ifstream file("input.txt");
    if(!file.is_open()) {
        cerr << "Error: Could not open input.txt" << endl;
        return 1;
    }

    string text((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();

    // Create vocabulary
    string unique_chars = text;
    sort(unique_chars.begin(), unique_chars.end());
    unique_chars.erase(unique(unique_chars.begin(), unique_chars.end()), unique_chars.end());
    
    vocab_size = unique_chars.size();
    
    // Create mappings
    for(size_t i = 0; i < unique_chars.size(); i++) {
        char_to_idx[unique_chars[i]] = i;
        idx_to_char[i] = unique_chars[i];
    }

    // Encode text
    vector<int> data = encode(text);

    // Train-test split
    int n = (int)(0.9 * data.size());
    train_data = vector<int>(data.begin(), data.begin() + n);
    val_data = vector<int>(data.begin() + n, data.end());

    cout << "Vocab size: " << vocab_size << endl;
    cout << "Train data size: " << train_data.size() << endl;
    cout << "Val data size: " << val_data.size() << endl;

    // Create model
    GPTLanguageModel model;
    
    // Load checkpoint if it exists
    if (load_checkpoint) {
        ifstream check_file(checkpoint_file);
        if (check_file.good()) {
            check_file.close();
            cout << "Loading checkpoint from " << checkpoint_file << "..." << endl;
            auto loaded_state_dict = load_safe_tensors(checkpoint_file);
            auto [missing_keys, unexpected_keys] = model.load_state_dict(loaded_state_dict, true);
            
            if (!missing_keys.empty()) {
                cout << "Warning: Missing keys when loading checkpoint:" << endl;
                for (const auto& key : missing_keys) {
                    cout << "  - " << key << endl;
                }
            }
            if (!unexpected_keys.empty()) {
                cout << "Warning: Unexpected keys when loading checkpoint:" << endl;
                for (const auto& key : unexpected_keys) {
                    cout << "  - " << key << endl;
                }
            }
            cout << "Checkpoint loaded successfully!" << endl;
        } else {
            cout << "No checkpoint found at " << checkpoint_file << ", starting from scratch" << endl;
        }
    }
    
    // Count parameters
    auto params = model.parameters();
    int total_params = 0;
    for(auto* p : params) {
        total_params += p->tensor().size();
    }
    cout << total_params / 1e6 << " M parameters" << endl;

    // Create optimizer
    AdamW optimizer(params, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f);

    // // Training loop
    // for(int iter = 0; iter < max_iters; iter++) {
    //     cout << "\n=== Iteration " << iter << " ===" << endl;
        
    //     // Evaluate
    //     if(iter % eval_interval == 0 || iter == max_iters - 1) {
    //         // cout << "[Main] Estimating loss..." << endl;
    //         auto losses = estimate_loss(model);
    //         cout << "step " << iter << ": train loss " << losses["train"] << ", val loss " << losses["val"] << endl;
    //     }

    //     // Save checkpoint if enabled
    //     if(save_checkpoint) {
    //         if(iter % save_iters == 0 || iter == max_iters - 1) {
    //             string checkpoint_name = "checkpoint_iter_" + to_string(iter) + ".safetensors";
    //             auto state_dict = model.state_dict();
    //             write_safe_tensors(state_dict, checkpoint_name);
    //         }
    //     }

    //     // Sample batch
    //     // cout << "[Main] Getting batch..." << endl;
    //     auto [xb, yb] = get_batch("train");
    //     // cout << "[Main] xb shape: [";
    //     // for(int i = 0; i < xb.shape().size(); i++) {
    //     //     cout << xb.shape()[i];
    //     //     if(i < xb.shape().size() - 1) cout << ", ";
    //     // }
    //     // cout << "]" << endl;
    //     // cout << "[Main] yb shape: [";
    //     // for(int i = 0; i < yb.shape().size(); i++) {
    //     //     cout << yb.shape()[i];
    //     //     if(i < yb.shape().size() - 1) cout << ", ";
    //     // }
    //     // cout << "]" << endl;

    //     // Forward pass
    //     // cout << "[Main] Starting forward pass..." << endl;
    //     try {
    //         auto [logits, loss] = model.forward_with_targets(xb, &yb);
    //         // cout << "[Main] Forward pass complete" << endl;

    //         // Backward pass
    //         // cout << "[Main] Starting backward pass..." << endl;
    //         optimizer.zero_grad();
    //         loss.set_grad(ones({1}));
    //         // cout << "[Main] Loss gradient: " << loss.grad().at(0) << endl;
    //         loss.backward();
    //         // cout << "[Main] Backward pass complete, calling optimizer.step()" << endl;
    //         optimizer.step();
    //         // cout << "[Main] Optimizer step complete" << endl;
    //     } catch(const exception& e) {
    //         cerr << "[Main] ERROR in iteration " << iter << ": " << e.what() << endl;
    //         throw;
    //     }
    // }

    // Generate - start with zeros context like PyTorch: torch.zeros((1, 1), dtype=torch.long)
    // Create a tensor with shape (1, 1) containing 0 (first token index)
    // cout << "[Main] Starting text generation..." << endl;
    Tensor context({1, 1}, {0.0f}, false);
    // Tensor generated = model.generate(context, 500);
    Tensor generated = model.generate(context, 10);
    
    // Decode and print - extract indices from generated tensor
    vector<int> generated_vec;
    int B = generated.shape()[0];
    int T = generated.shape()[1];
    for(int b = 0; b < B; b++) {
        for(int t = 0; t < T; t++) {
            vector<int> idx = {b, t};
            generated_vec.push_back((int)generated.at(idx));
        }
    }
    string output = decode(generated_vec);
    cout << output << endl;

    // Write longer generation to more.txt
    ofstream outfile("more.txt");
    if(!outfile.is_open()) {
        cerr << "Error: Could not open more.txt for writing" << endl;
        return 1;
    }
    
    // Reset context for longer generation
    Tensor context_long({1, 1}, {0.0f}, false);
    // Tensor generated_long = model.generate(context_long, 10000);
    Tensor generated_long = model.generate(context_long, 50);
    
    vector<int> generated_long_vec;
    B = generated_long.shape()[0];
    T = generated_long.shape()[1];
    for(int b = 0; b < B; b++) {
        for(int t = 0; t < T; t++) {
            vector<int> idx = {b, t};
            generated_long_vec.push_back((int)generated_long.at(idx));
        }
    }
    outfile << decode(generated_long_vec);
    outfile.close();
    
    cout << "\nGenerated text written to more.txt" << endl;

    return 0;
}

