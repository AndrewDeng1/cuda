#ifndef NN_H
#define NN_H

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <type_traits>

// Forward declarations
class Module;

// Parameter: a wrapper around Tensor that marks it as a learnable parameter
class Parameter {
    Tensor data_;

public:
    Parameter() = default;
    explicit Parameter(Tensor data) : data_(std::move(data)) {
        data_.set_requires_grad(true);
    }

    Tensor& tensor() { return data_; }
    const Tensor& tensor() const { return data_; }
};

// Module: base class for all neural network modules
class Module {
protected:
    // Lazy registration flag - set false on move, triggers re-registration on access
    mutable bool registered_ = false;
    
    // Owned storage for dynamically added modules (used by ModuleList/Sequential)
    std::vector<std::unique_ptr<Module>> owned_modules_;
    
    // Ensures registration is done (calls virtual methods after move)
    void ensure_registered() const {
        if (!registered_) {
            auto* self = const_cast<Module*>(this);
            self->parameters_.clear();
            self->submodules_.clear();
            self->register_params();
            self->register_submodules();
            registered_ = true;
        }
    }

public:
    // name -> parameter (non-owning pointers)
    std::unordered_map<std::string, Parameter*> parameters_;

    // name -> child module (non-owning pointers)
    std::unordered_map<std::string, Module*> submodules_;

    // Training mode flag
    bool training_ = true;

    virtual ~Module() = default;
    
    Module() : registered_(false) {}  // Will register on first access
    
    // Delete copy
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    
    // Move - invalidate registration, will re-register on first access
    Module(Module&& other) noexcept 
        : registered_(false), 
          owned_modules_(std::move(other.owned_modules_)),
          parameters_(), 
          submodules_(), 
          training_(other.training_) {}
    Module& operator=(Module&& other) noexcept {
        if (this != &other) {
            registered_ = false;
            owned_modules_ = std::move(other.owned_modules_);
            parameters_.clear();
            submodules_.clear();
            training_ = other.training_;
        }
        return *this;
    }

    virtual Tensor forward(const Tensor& x) = 0;

    // Operator() for convenience
    Tensor operator()(const Tensor& x) { return forward(x); }

    // Low-level registration
    void register_param(const std::string& name, Parameter* p) {
        parameters_[name] = p;
    }

    void register_module(const std::string& name, Module* m) {
        submodules_[name] = m;
    }

    // Override these in subclasses to register members
    // Called automatically on first access after construction/move
    virtual void register_params() {}
    virtual void register_submodules() {}

    // Recursive parameter collection
    std::vector<Parameter*> parameters() {
        ensure_registered();
        std::vector<Parameter*> out;
        collect_parameters(out);
        return out;
    }

    void collect_parameters(std::vector<Parameter*>& out) {
        ensure_registered();  // Ensure this module is registered
        for (auto& [_, p] : parameters_)
            out.push_back(p);

        for (auto& [_, m] : submodules_)
            m->collect_parameters(out);  // Recursively collects (and ensures registration)
    }

    // Training/eval mode
    void train(bool mode = true) {
        ensure_registered();
        training_ = mode;
        for (auto& [_, m] : submodules_)
            m->train(mode);
    }

    void eval() { train(false); }

    bool is_training() const { return training_; }
};

// Linear layer: y = xW^T + b
class Linear : public Module {
public:
    int in_features_;
    int out_features_;
    bool use_bias_;
    Parameter weight_;
    Parameter bias_;

    Linear(int in_features, int out_features, bool bias = true);
    
    // Default move works - lazy registration handles re-registration
    Linear(Linear&&) = default;
    Linear& operator=(Linear&&) = default;
    
    Tensor forward(const Tensor& x) override;
    void register_params() override;
};

// Dropout layer
class Dropout : public Module {
public:
    float p_;

    explicit Dropout(float p = 0.5f);
    
    Dropout(Dropout&&) = default;
    Dropout& operator=(Dropout&&) = default;
    
    Tensor forward(const Tensor& x) override;
    // No parameters or submodules - default empty implementations
};

// LayerNorm layer
class LayerNorm : public Module {
public:
    int normalized_shape_;
    float eps_;
    Parameter gamma_;
    Parameter beta_;

    explicit LayerNorm(int normalized_shape, float eps = 1e-5f);
    
    LayerNorm(LayerNorm&&) = default;
    LayerNorm& operator=(LayerNorm&&) = default;
    
    Tensor forward(const Tensor& x) override;
    void register_params() override;
};

// Embedding layer
class Embedding : public Module {
public:
    int num_embeddings_;
    int embedding_dim_;
    Parameter weight_;

    Embedding(int num_embeddings, int embedding_dim);
    
    Embedding(Embedding&&) = default;
    Embedding& operator=(Embedding&&) = default;
    
    Tensor forward(const Tensor& indices) override;
    void register_params() override;
};

// ModuleList: holds a list of modules (uses inherited owned_modules_ for storage)
// NOTE: Ability to remove Module from ModuleList not implemented, not necessary for now
class ModuleList : public Module {
public:
    ModuleList() = default;
    explicit ModuleList(std::vector<std::unique_ptr<Module>> modules);
    
    ModuleList(ModuleList&&) = default;
    ModuleList& operator=(ModuleList&&) = default;

    template<typename T>
    void append(T module) {
        static_assert(std::is_base_of_v<Module, T>, "T must derive from Module");
        owned_modules_.push_back(std::make_unique<T>(std::move(module)));
        registered_ = false;  // Invalidate to re-register on next access
    }
    
    void append(std::unique_ptr<Module> m);
    
    Module* operator[](size_t idx) { return owned_modules_[idx].get(); }
    size_t size() const { return owned_modules_.size(); }

    Tensor forward(const Tensor& x) override {
        throw std::runtime_error("ModuleList has no forward method. Access modules directly.");
    }
    
    void register_submodules() override;
};

// Sequential: applies modules in sequence (uses inherited owned_modules_ for storage)
// NOTE: Ability to remove Module from Sequential not implemented, not necessary for now
class Sequential : public Module {
public:
    Sequential() = default;
    explicit Sequential(std::vector<std::unique_ptr<Module>> modules);
    
    Sequential(Sequential&&) = default;
    Sequential& operator=(Sequential&&) = default;

    template<typename T>
    void append(T module) {
        static_assert(std::is_base_of_v<Module, T>, "T must derive from Module");
        owned_modules_.push_back(std::make_unique<T>(std::move(module)));
        registered_ = false;  // Invalidate to re-register on next access
    }
    
    void append(std::unique_ptr<Module> m);
    
    Tensor forward(const Tensor& x) override;
    void register_submodules() override;
};

#endif // NN_H
