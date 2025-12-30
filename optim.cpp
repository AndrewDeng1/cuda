#include "optim.h"
#include "nn.h"
#include <cmath>

// ============================================================================
// Optimizer
// ============================================================================

Optimizer::Optimizer(std::vector<Parameter*> parameters, std::unordered_map<std::string, float> default_options) {
    param_groups_.emplace_back(std::move(parameters), std::move(default_options));
}

Optimizer::Optimizer(std::vector<ParamGroup> param_groups) 
    : param_groups_(std::move(param_groups)) {}

void Optimizer::zero_grad() {
    for (auto& group : param_groups_) {
        for (auto* param : group.params) {
            if (param && param->tensor().has_grad()) {
                Tensor grad = param->tensor().grad();
                // Zero out the gradient tensor
                for (int i = 0; i < grad.size(); i++) {
                    grad.at(i) = 0.0f;
                }
            }
        }
    }
}

// ============================================================================
// SGD
// ============================================================================

SGD::SGD(std::vector<Parameter*> parameters, float lr, float momentum, float weight_decay)
    : Optimizer(std::vector<ParamGroup>{}) {
    std::unordered_map<std::string, float> options;
    options["lr"] = lr;
    options["momentum"] = momentum;
    options["weight_decay"] = weight_decay;
    param_groups_.emplace_back(std::move(parameters), std::move(options));
}

SGD::SGD(std::vector<ParamGroup> param_groups) : Optimizer(std::move(param_groups)) {
    // Ensure all groups have required options with defaults
    for (auto& group : param_groups_) {
        if (group.options.find("lr") == group.options.end()) {
            group.options["lr"] = 0.01f;
        }
        if (group.options.find("momentum") == group.options.end()) {
            group.options["momentum"] = 0.0f;
        }
        if (group.options.find("weight_decay") == group.options.end()) {
            group.options["weight_decay"] = 0.0f;
        }
    }
}

void SGD::step() {
    for (auto& group : param_groups_) {
        float lr = group.options.at("lr");
        float momentum = group.options.at("momentum");
        float weight_decay = group.options.at("weight_decay");
        
        for (auto* param : group.params) {
            if (!param || !param->tensor().has_grad()) continue;
            
            Tensor& param_tensor = param->tensor();
            Tensor grad = param_tensor.grad();
            
            // Apply weight decay
            if (weight_decay > 0.0f) {
                grad = grad + param_tensor * weight_decay;
            }
            
            // Initialize or get momentum buffer
            if (momentum > 0.0f) {
                // Create state if it doesn't exist
                if (state_.find(param) == state_.end()) {
                    state_[param] = OptimizerState();
                }
                
                // Create momentum buffer if it doesn't exist
                if (state_[param].buffers.find("momentum") == state_[param].buffers.end()) {
                    state_[param].buffers["momentum"] = Tensor(param_tensor.shape(), 0.0f);
                }
                
                Tensor& momentum_buf = state_[param].buffers["momentum"];
                
                // Update momentum buffer: buf = momentum * buf + grad
                momentum_buf = momentum_buf * momentum + grad;
                
                // Update parameter: param = param - lr * momentum_buf
                param_tensor = param_tensor - momentum_buf * lr;
            } else {
                // No momentum: param = param - lr * grad
                param_tensor = param_tensor - grad * lr;
            }
        }
    }
}

// ============================================================================
// Adam
// ============================================================================

Adam::Adam(std::vector<Parameter*> parameters, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::vector<ParamGroup>{}) {
    std::unordered_map<std::string, float> options;
    options["lr"] = lr;
    options["beta1"] = beta1;
    options["beta2"] = beta2;
    options["eps"] = eps;
    options["weight_decay"] = weight_decay;
    param_groups_.emplace_back(std::move(parameters), std::move(options));
}

Adam::Adam(std::vector<ParamGroup> param_groups) : Optimizer(std::move(param_groups)) {
    // Ensure all groups have required options with defaults
    for (auto& group : param_groups_) {
        if (group.options.find("lr") == group.options.end()) {
            group.options["lr"] = 0.001f;
        }
        if (group.options.find("beta1") == group.options.end()) {
            group.options["beta1"] = 0.9f;
        }
        if (group.options.find("beta2") == group.options.end()) {
            group.options["beta2"] = 0.999f;
        }
        if (group.options.find("eps") == group.options.end()) {
            group.options["eps"] = 1e-8f;
        }
        if (group.options.find("weight_decay") == group.options.end()) {
            group.options["weight_decay"] = 0.0f;
        }
    }
}

void Adam::step() {
    for (auto& group : param_groups_) {
        float lr = group.options.at("lr");
        float beta1 = group.options.at("beta1");
        float beta2 = group.options.at("beta2");
        float eps = group.options.at("eps");
        float weight_decay = group.options.at("weight_decay");
        
        for (auto* param : group.params) {
            if (!param || !param->tensor().has_grad()) continue;
            
            Tensor& param_tensor = param->tensor();
            Tensor grad = param_tensor.grad();
            
            // Apply weight decay
            if (weight_decay > 0.0f) {
                grad = grad + param_tensor * weight_decay;
            }
            
            // Initialize state if needed
            if (state_.find(param) == state_.end()) {
                state_[param] = OptimizerState();
            }
            OptimizerState& state = state_[param];
            
            // Initialize buffers if needed
            if (state.buffers.find("exp_avg") == state.buffers.end()) {
                state.buffers["exp_avg"] = Tensor(param_tensor.shape(), 0.0f);
            }
            if (state.buffers.find("exp_avg_sq") == state.buffers.end()) {
                state.buffers["exp_avg_sq"] = Tensor(param_tensor.shape(), 0.0f);
            }
            
            Tensor& exp_avg = state.buffers["exp_avg"];
            Tensor& exp_avg_sq = state.buffers["exp_avg_sq"];
            state.step_count++;
            
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
            exp_avg = exp_avg * beta1 + grad * (1.0f - beta1);
            
            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
            Tensor grad_sq = grad * grad;
            exp_avg_sq = exp_avg_sq * beta2 + grad_sq * (1.0f - beta2);
            
            // Compute bias-corrected estimates
            float bias_correction1 = 1.0f - std::pow(beta1, state.step_count);
            float bias_correction2 = 1.0f - std::pow(beta2, state.step_count);
            
            Tensor m_hat = exp_avg / bias_correction1;
            Tensor v_hat = exp_avg_sq / bias_correction2;
            
            // Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            Tensor v_hat_sqrt = v_hat.pow(0.5f);
            param_tensor = param_tensor - m_hat * lr / (v_hat_sqrt + eps);
        }
    }
}

// ============================================================================
// AdamW
// ============================================================================

AdamW::AdamW(std::vector<Parameter*> parameters, float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::vector<ParamGroup>{}) {
    std::unordered_map<std::string, float> options;
    options["lr"] = lr;
    options["beta1"] = beta1;
    options["beta2"] = beta2;
    options["eps"] = eps;
    options["weight_decay"] = weight_decay;
    param_groups_.emplace_back(std::move(parameters), std::move(options));
}

AdamW::AdamW(std::vector<ParamGroup> param_groups) : Optimizer(std::move(param_groups)) {
    // Ensure all groups have required options with defaults
    for (auto& group : param_groups_) {
        if (group.options.find("lr") == group.options.end()) {
            group.options["lr"] = 0.001f;
        }
        if (group.options.find("beta1") == group.options.end()) {
            group.options["beta1"] = 0.9f;
        }
        if (group.options.find("beta2") == group.options.end()) {
            group.options["beta2"] = 0.999f;
        }
        if (group.options.find("eps") == group.options.end()) {
            group.options["eps"] = 1e-8f;
        }
        if (group.options.find("weight_decay") == group.options.end()) {
            group.options["weight_decay"] = 0.01f;
        }
    }
}

void AdamW::step() {
    for (auto& group : param_groups_) {
        float lr = group.options.at("lr");
        float beta1 = group.options.at("beta1");
        float beta2 = group.options.at("beta2");
        float eps = group.options.at("eps");
        float weight_decay = group.options.at("weight_decay");
        
        for (auto* param : group.params) {
            if (!param || !param->tensor().has_grad()) continue;
            
            Tensor& param_tensor = param->tensor();
            Tensor grad = param_tensor.grad();
            
            // Note: AdamW does NOT apply weight decay to gradients
            // Weight decay is applied directly to parameters (decoupled)
            
            // Initialize state if needed
            if (state_.find(param) == state_.end()) {
                state_[param] = OptimizerState();
            }
            OptimizerState& state = state_[param];
            
            // Initialize buffers if needed
            if (state.buffers.find("exp_avg") == state.buffers.end()) {
                state.buffers["exp_avg"] = Tensor(param_tensor.shape(), 0.0f);
            }
            if (state.buffers.find("exp_avg_sq") == state.buffers.end()) {
                state.buffers["exp_avg_sq"] = Tensor(param_tensor.shape(), 0.0f);
            }
            
            Tensor& exp_avg = state.buffers["exp_avg"];
            Tensor& exp_avg_sq = state.buffers["exp_avg_sq"];
            state.step_count++;
            
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
            exp_avg = exp_avg * beta1 + grad * (1.0f - beta1);
            
            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
            Tensor grad_sq = grad * grad;
            exp_avg_sq = exp_avg_sq * beta2 + grad_sq * (1.0f - beta2);
            
            // Compute bias-corrected estimates
            float bias_correction1 = 1.0f - std::pow(beta1, state.step_count);
            float bias_correction2 = 1.0f - std::pow(beta2, state.step_count);
            
            Tensor m_hat = exp_avg / bias_correction1;
            Tensor v_hat = exp_avg_sq / bias_correction2;
            
            // Update parameter: param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
            // This is the decoupled weight decay - applied directly to parameters, not gradients
            Tensor v_hat_sqrt = v_hat.pow(0.5f);
            param_tensor = param_tensor - lr * (m_hat / (v_hat_sqrt + eps) + param_tensor * weight_decay);
        }
    }
}

