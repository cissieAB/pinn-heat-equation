// Copyright 2021, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

/*
Created by Xinxin Mei on 7/29/22.

Try to implement the heat/diffusion equation defined in ../pde_solver with cpp PINN.
The Jupyter code is at https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb

References:
- Pytorch official tutorial:
https://pytorch.org/tutorials/advanced/cpp_frontend.html
https://github.com/pytorch/examples/tree/main/cpp
- Online tutorials:
https://tebesu.github.io/posts/PyTorch-C++-Frontend
https://medium.com/pytorch/a-taste-of-pytorch-c-frontend-api-8ec5209823ca

Python code change is ignored in this branch.
*/

#include <torch/torch.h>
#include <math.h>
#include <iostream>

using namespace torch::indexing;   // for tensor indexing

constexpr float PI = 3.14159265358979323846;

const int N = 65;  // grid dimension
const float STEP_SIZE = 1.0 / (N - 1);
const int WHOLE_GRID_SIZE = N * N;
const int WHOLE_INPUT_DATA_SIZE = WHOLE_GRID_SIZE * 2;
const int BD_SIZE = 4 * N;
const int BD_INPUT_SIZE = BD_SIZE * 2;

const int ADAM_STEPS = 1000;

// max step is set based on experience. 5000 steps make loss at e-5 level.
const int MAX_STEPS = 5000;
// const int MAX_STEPS = 1;  // from ncu compiling
// criteria for stop training.
// When loss is at 1.x~e-5, it will stop degrading even the training continues
const float TARGET_LOSS = 5.0e-5;

const int NN_INPUT_SIZE = 2;
const int NN_OUTPUT_SIZE = 1;
const int NN_HIDDEN_SIZE = 20;

/// NN structure declaration
struct HeatPINNNetImpl: torch::nn::Module {
    /**
     * Declare the NN to match the Python code in
     * https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb
     * NN(
          (layers): Sequential(
            (input): Linear(in_features=2, out_features=20, bias=True)
            (input_activation): Tanh()
            (hidden_0): Linear(in_features=20, out_features=20, bias=True)
            (activation_0): Tanh()
            (hidden_1): Linear(in_features=20, out_features=20, bias=True)
            (activation_1): Tanh()
            (hidden_2): Linear(in_features=20, out_features=20, bias=True)
            (activation_2): Tanh()
            (hidden_3): Linear(in_features=20, out_features=20, bias=True)
            (activation_3): Tanh()
            (output): Linear(in_features=20, out_features=1, bias=True)
          )
        )
     */

    // TODO: manually declare 4 hidden layers first. Add loop or use torch::OrderedDict later
    HeatPINNNetImpl(int input_layer_size, int output_layer_size, int hidden_layer_size)
            :
            input(torch::nn::Linear(input_layer_size, hidden_layer_size)),
            hidden_0(torch::nn::Linear(hidden_layer_size, hidden_layer_size)),
            hidden_1(torch::nn::Linear(hidden_layer_size, hidden_layer_size)),
            hidden_2(torch::nn::Linear(hidden_layer_size, hidden_layer_size)),
            hidden_3(torch::nn::Linear(hidden_layer_size, hidden_layer_size)),
            output(torch::nn::Linear(hidden_layer_size, output_layer_size))

    {
        // register module
        register_module("input", input);
        register_module("hidden_0", hidden_0);
        register_module("hidden_1", hidden_1);
        register_module("hidden_2", hidden_2);
        register_module("hidden_3", hidden_3);
        register_module("output", output);
    }

    torch::Tensor forward(torch::Tensor x) {
        // activation function for the input and hidden layers
        x = torch::tanh(input(x));
        x = torch::tanh(hidden_0(x));
        x = torch::tanh(hidden_1(x));
        x = torch::tanh(hidden_2(x));
        x = torch::tanh(hidden_3(x));
        x = output(x);

        return x;
    }

    torch::nn::Linear input, hidden_0, hidden_1, hidden_2, hidden_3, output;
};

TORCH_MODULE(HeatPINNNet);  //? a wrapped shared_ptr, see official tutorial

void get_whole_dataset_X(float* data) {
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) {
            int idx_base = 2 * (ix * N + iy);
            data[idx_base] = ix * STEP_SIZE;
            data[idx_base + 1] = iy * STEP_SIZE;
        }
}

void get_bc_dataset_xTrain(float* data) {
    for (int i = 0; i < N; i++) {
        int idx_base = 2 * i;
        float num = i * STEP_SIZE;
        // x: left, right, down, top
        data[idx_base] = 0.0;
        data[idx_base + 2 * N] = 1.0;
        data[idx_base + 4 * N] = num;
        data[idx_base + 6 * N] = num;

        // y: left, right, down, top
        idx_base += 1;
        data[idx_base] = num;
        data[idx_base + 2 * N] = num;
        data[idx_base + 4 * N] = 0.0;
        data[idx_base + 6 * N] = 1.0;
    }
}

float get_pde_f_term(float x, float y) {
    /**
     * Get the f term of the PDE.
     * f = -2 * pi * pi * sin(pi * x) * sin(pi * y)
     */
    return sin(PI * x) * sin(PI * y);
}

void get_fterm_dataset_f(float* data) {
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) {
            data[ix * N + iy] = get_pde_f_term(ix * STEP_SIZE, iy * STEP_SIZE);
        }
}

torch::Tensor get_pde_loss(torch::Tensor& u, torch::Tensor& X, torch::Device& device) {
    /**
     * Get the pde loss based on the NN forward results.
     * Calculate the gradients and the pde terms.
     */

    // get the gradients
    torch::Tensor du_dX = torch::autograd::grad(
            /*output=*/{u},
            /*input=*/{X},
            /*grad_outputs=*/{torch::ones_like(u)},
            /*retain_graph=*/true,
            /*create_graph=*/true,
            /*allow_unused=*/true)[0];

    torch::Tensor du_dx = du_dX.index({"...", 0});
    torch::Tensor du_dy = du_dX.index({"...", 1});

    torch::Tensor du_dxx = torch::autograd::grad({du_dx}, {X}, {torch::ones_like(du_dx),},
                                                 true, true, true)[0].index({"...", 0});
    torch::Tensor du_dyy = torch::autograd::grad({du_dy}, {X}, {torch::ones_like(du_dy),},
                                                 true, true, true)[0].index({"...", 1});
//    std::cout << "du_dxx + du_dyy:\n" << du_dxx + du_dyy << std::endl;

    // get constant term f_X
    float f_data[WHOLE_GRID_SIZE];
    get_fterm_dataset_f(f_data);
    torch::Tensor f_X = -2.0 * PI * PI * torch::from_blob(f_data, {WHOLE_GRID_SIZE}).to(device);

    return torch::mse_loss(du_dxx + du_dyy, f_X);
}

torch::Tensor get_total_loss(
        HeatPINNNet& net,
        torch::Tensor& X, torch::Tensor& X_train, torch::Tensor& y_train,
	torch::Device& device
        ) {
    /**
     * Calculate the loss of each step.
     * loss_train is from the training dataset. loss_pde is from the whole dataset.
     */
    torch::Tensor u = net->forward(X);

    return torch::mse_loss(net->forward(X_train), y_train) + get_pde_loss(u, X, device);
}

int main() {

    std::cout << "####### A cpp torch example with PINN heat equation. #######\n" << std::endl;

    /**
     * Init NN structure.
     */
    // Device
    auto cuda_available = torch::cuda::is_available();
    auto device_str = cuda_available ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_str);
    std::cout << (cuda_available ? "CUDA available. Training on GPU.\n" : "Training on CPU.\n") << '\n';

    auto net = HeatPINNNet(NN_INPUT_SIZE, NN_OUTPUT_SIZE, NN_HIDDEN_SIZE);  // init a network model
    net->to(device);

    /**
     * Init data sets.
     */
    // supervised training data set
    torch::Tensor y_train, X_train, X;
    // TODO: seems must choose kFloat32 data type now because of the NN declaration. Check later.
    y_train = torch::zeros({BD_SIZE, NN_OUTPUT_SIZE}, device);
    std::cout << "y_train sizes: " << y_train.sizes() << std::endl;
    std::cout << "y_train.device().type(): " << y_train.device().type() << std::endl;
    std::cout << "y_train.requires_grad(): " << y_train.requires_grad() << std::endl;


    float X_train_data[BD_INPUT_SIZE];
    get_bc_dataset_xTrain(X_train_data);
    X_train = torch::from_blob(X_train_data, {BD_SIZE, NN_INPUT_SIZE}).to(device);
    //X_train = torch::from_blob(X_train_data, {BD_SIZE, NN_INPUT_SIZE}, options);
    std::cout << "X_train sizes: " << X_train.sizes() << std::endl;
    std::cout << "X_train.device().type(): " << X_train.device().type() << std::endl;
    std::cout << "X_train.requires_grad(): " << X_train.requires_grad() << std::endl;

    // whole data set
    float X_data[WHOLE_INPUT_DATA_SIZE];
    get_whole_dataset_X(X_data);
    X = torch::from_blob(X_data, {WHOLE_GRID_SIZE, NN_INPUT_SIZE}, torch::requires_grad()).to(device);
    std::cout << "X sizes: " << X.sizes() << std::endl;
    std::cout << "X.device().index(): " << X.device().index() << std::endl;
    std::cout << "X.requires_grad(): " << X.requires_grad() << std::endl;

    /*
     * Training process
     *  The training steps are trying to match the Python script.
     *  First 1000 steps use Adam, and remaining steps use LBFGS.
     */

    std::cout << "\n\nTraining started..." << std::endl;

    torch::Tensor loss_sum;
    int iter = 1;

    // optimizer declaration. All parameters are trying to match Python
    torch::optim::Adam adam_optim(net->parameters(), torch::optim::AdamOptions(1e-3));  // default Adam lr
    // Python default value ref: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
    torch::optim::LBFGSOptions LBFGS_optim_options =
            torch::optim::LBFGSOptions(1).max_iter(50000).max_eval(50000).history_size(50);
    torch::optim::LBFGS LBFGS_optim(net->parameters(), LBFGS_optim_options);

    while (iter <= MAX_STEPS) {
        auto closure = [&]() {
            LBFGS_optim.zero_grad();
            loss_sum = get_total_loss(net, X, X_train, y_train, device);
            loss_sum.backward();
            return loss_sum;
        };

        if (iter <= ADAM_STEPS) {
            adam_optim.step(closure);
        } else {
            LBFGS_optim.step(closure);
        }

        // print loss info
        if (iter % 100 == 0) {
            std::cout << "  iter=" << iter << ", loss=" << std::setprecision(7) << loss_sum.item<float>();
	    std::cout << ", loss.device().type()=" << loss_sum.device().type() << std::endl;
        }

        // stop training
        if (loss_sum.item<float>() < TARGET_LOSS) {
            iter += 1;
            break;
	}

        iter += 1;
    }

    std::cout << "\nTraining stopped." << std::endl;
    std::cout << "Final iter=" << iter - 1 << ", loss=" << std::setprecision(7) << loss_sum.item<float>();
    std::cout << ", loss.device().type()=" << loss_sum.device().type() << std::endl;

    return 0;
}
