#ifndef OPTIM_H
#define OPTIM_H

#include "nn.h"
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declaration
class Parameter;

// Parameter group - contains parameters and their options
struct ParamGroup {
    std::vector<Parameter*> params;
    std::unordered_map<std::string, float> options;  // e.g., {"lr": 0.01, "momentum": 0.9}
    
    ParamGroup(std::vector<Parameter*> params, std::unordered_map<std::string, float> options = {})
        : params(std::move(params)), options(std::move(options)) {}
};

// Optimizer state - stores optimizer-specific state per parameter
// For SGD: momentum buffer
// For Adam: exp_avg, exp_avg_sq, step
struct OptimizerState {
    std::unordered_map<std::string, Tensor> buffers;  // e.g., {"momentum": Tensor(...)}
    int step_count = 0;
};

// Base Optimizer class
class Optimizer {
protected:
    std::vector<ParamGroup> param_groups_;
    std::unordered_map<Parameter*, OptimizerState> state_;

public:
    Optimizer(std::vector<Parameter*> parameters, std::unordered_map<std::string, float> default_options = {});
    Optimizer(std::vector<ParamGroup> param_groups);
    
    virtual ~Optimizer() = default;
    
    // Zero all gradients
    void zero_grad();
    
    // Perform optimization step (to be overridden)
    virtual void step() = 0;
    
    // Accessors
    const std::vector<ParamGroup>& param_groups() const { return param_groups_; }
    std::vector<ParamGroup>& param_groups() { return param_groups_; }
    const std::unordered_map<Parameter*, OptimizerState>& state() const { return state_; }
    std::unordered_map<Parameter*, OptimizerState>& state() { return state_; }
};

// SGD Optimizer
class SGD : public Optimizer {
public:
    SGD(std::vector<Parameter*> parameters, float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    SGD(std::vector<ParamGroup> param_groups);
    
    void step() override;
};

// Adam Optimizer
class Adam : public Optimizer {
public:
    Adam(std::vector<Parameter*> parameters, 
         float lr = 0.001f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f, 
         float eps = 1e-8f,
         float weight_decay = 0.0f);
    Adam(std::vector<ParamGroup> param_groups);
    
    void step() override;
};

// AdamW Optimizer (Adam with decoupled weight decay)
class AdamW : public Optimizer {
public:
    AdamW(std::vector<Parameter*> parameters, 
          float lr = 0.001f, 
          float beta1 = 0.9f, 
          float beta2 = 0.999f, 
          float eps = 1e-8f,
          float weight_decay = 0.01f);
    AdamW(std::vector<ParamGroup> param_groups);
    
    void step() override;
};

#endif // OPTIM_H

