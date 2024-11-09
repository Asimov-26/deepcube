#include <iostream>
#include <string>
#include <fstream>
#include "cube.h"
#include "nn_math.h"
#include "nn_cost.h"
#include "nn_layer.h"
#include <vector> 

const int n_epoch = 1000;
const int n_cube = 100;
const int n_scramble = 25;
const double lr = 1.0 / (n_cube * n_scramble);
const std::string save_dir = "/tmp/";

int main() {
  Cube cube;
  
  // Prepare network
  InputLayer input_layer(state_size);
  DenseLayer dense_layer1(512, &input_layer);
  dense_layer1.init_params();
  ReluLayer relu_layer1(&dense_layer1);
  DenseLayer dense_layer2(256, &relu_layer1);
  dense_layer2.init_params();
  ReluLayer relu_layer2(&dense_layer2);
  
  DenseLayer dense_layer_p1(256, &relu_layer2);
  dense_layer_p1.init_params();
  ReluLayer relu_layer_p1(&dense_layer_p1);
  DenseLayer dense_layer_p2(n_move, &relu_layer_p1);
  dense_layer_p2.init_params();
  
  DenseLayer dense_layer_v1(256, &relu_layer2);
  dense_layer_v1.init_params();
  ReluLayer relu_layer_v1(&dense_layer_v1);
  DenseLayer dense_layer_v2(1, &relu_layer_v1);
  dense_layer_v2.init_params();
  
  double cost_target, cost_v, cost_p;
  double sample_weight;
  double target_value;
  int target_idx;
  // Include vector header for dynamic arrays

  std::vector<double> squared_losses(n_epoch, 0.0);  // Stores squared losses for each epoch
  std::vector<double> cross_entropy_losses(n_epoch, 0.0); 
  std::vector<double> target_costs(n_epoch, 0.0); 
  
  double *const values = new double[n_move];
  
std::cout << "Total no of Epochs " <<n_epoch<< std::endl;
for (int epoch = 0; epoch < n_epoch; ++epoch) {
    cost_target = 0.0;
    cost_v = 0.0;
    cost_p = 0.0;

    for (int i = 0; i < n_cube; ++i) {
        cube.init();
        for (int j = 0; j < n_scramble; ++j) {
            cube.rotate_random();
            sample_weight = 1.0 / (j + 1);

            for (int k = 0; k < n_move; ++k) {
                Move cur_hypo_move = static_cast<Move>(k);
                cube.get_state_hypo(cur_hypo_move, input_layer.activations);
                dense_layer_v2.forward();

                if (cube.is_solved_hypo(cur_hypo_move)) {
                    values[k] = 1.0;
                } else {
                    values[k] = dense_layer_v2.activations[0] - 1.0;
                }
                dense_layer_v2.zero_states();
            }
            calc_max(values, n_move, &target_value, &target_idx);

            // Calculate gradients for value
            cube.get_state(input_layer.activations);
            dense_layer_v2.forward();
            cost_v += squared_error_grad(
                dense_layer_v2.activations, target_value, dense_layer_v2.feedbacks);
            cost_target += sample_weight * cost_v;
            dense_layer_v2.backward(sample_weight);
            dense_layer_v2.zero_states();

            // Calculate gradients for policy
            cube.get_state(input_layer.activations);
            dense_layer_p2.forward();
            cost_p += cross_entropy_loss_grad(dense_layer_p2.activations,
                target_idx, n_move, dense_layer_p2.feedbacks);
            cost_target += sample_weight * cost_p;
            dense_layer_p2.backward(sample_weight);
            dense_layer_p2.zero_states();
        }
    }

    // Apply gradients
    dense_layer2.apply_grad(lr);
    dense_layer2.zero_grad();

    cost_target /= n_cube;
    cost_v /= n_cube * n_scramble;
    cost_p /= n_cube * n_scramble;

    // Store the losses in the vectors for later use
    squared_losses[epoch] = cost_v;
    cross_entropy_losses[epoch] = cost_p;
    target_costs[epoch]=cost_target;

    std::cout << "Epoch " << epoch + 1 << std::endl;
    std::cout << "-- Target cost  " << cost_target << std::endl;
    std::cout << "-- Squared loss " << cost_v << std::endl;
    std::cout << "-- Cross entropy loss " << cost_p << std::endl;

    if (epoch == 0 || (epoch + 1) % 100 == 0) {
        dense_layer1.save(save_dir, "dense_layer1");
        dense_layer2.save(save_dir, "dense_layer2");
        dense_layer_p1.save(save_dir, "dense_layer_p1");
        dense_layer_p2.save(save_dir, "dense_layer_p2");
        dense_layer_v1.save(save_dir, "dense_layer_v1");
        dense_layer_v2.save(save_dir, "dense_layer_v2");
    }
}

delete[] values;

// Save the metrics to a CSV file
  std::ofstream file("metrics.csv");
  file << "Epoch,Target Cost,Squared Loss,Cross Entropy Loss\n";
  for (int epoch = 0; epoch < n_epoch; ++epoch) {
    file << epoch + 1 << "," << target_costs[epoch] << "," << squared_losses[epoch] << "," << cross_entropy_losses[epoch] << "\n";
}
file.close();

      // --------------------- Evaluation ---------------------

    // Evaluate the model on test cubes
    double test_cost_target = 0.0;
    double test_cost_v = 0.0;
    double test_cost_p = 0.0;
    int correct_policy_predictions = 0;

    for (int i = 0; i < n_test; ++i) {
        cube.init();
        for (int j = 0; j < n_scramble; ++j) {
            cube.rotate_random();
            
            // Get value prediction
            for (int k = 0; k < n_move; ++k) {
                Move cur_hypo_move = static_cast<Move>(k);
                cube.get_state_hypo(cur_hypo_move, input_layer.activations);
                dense_layer_v2.forward();

                if (cube.is_solved_hypo(cur_hypo_move)) {
                    values[k] = 1.0;
                } else {
                    values[k] = dense_layer_v2.activations[0] - 1.0;
                }
                dense_layer_v2.zero_states();
            }

            calc_max(values, n_move, &target_value, &target_idx);

            // Calculate value cost (squared error)
            cube.get_state(input_layer.activations);
            dense_layer_v2.forward();
            test_cost_v += squared_error_grad(dense_layer_v2.activations, target_value, dense_layer_v2.feedbacks);

            // Calculate policy cost (cross-entropy loss)
            cube.get_state(input_layer.activations);
            dense_layer_p2.forward();
            test_cost_p += cross_entropy_loss_grad(dense_layer_p2.activations, target_idx, n_move, dense_layer_p2.feedbacks);
            dense_layer_p2.zero_states();
            
            // Count correct policy predictions (matching predicted move)
            if (target_idx == std::distance(dense_layer_p2.activations, std::max_element(dense_layer_p2.activations, dense_layer_p2.activations + n_move))) {
                ++correct_policy_predictions;
            }
        }
    }

    test_cost_target = test_cost_v + test_cost_p;
    test_cost_v /= n_test * n_scramble;
    test_cost_p /= n_test * n_scramble;

    std::cout << "Evaluation results after " << n_epoch << " epochs:" << std::endl;
    std::cout << "-- Test Target cost: " << test_cost_target << std::endl;
    std::cout << "-- Test Squared loss: " << test_cost_v << std::endl;
    std::cout << "-- Test Cross entropy loss: " << test_cost_p << std::endl;
    std::cout << "-- Correct policy predictions: " << correct_policy_predictions << "/" << n_test * n_scramble << std::endl;

    // Save evaluation results to a file for later visualization
    std::ofstream eval_file("evaluation_results.csv");
    eval_file << "Test,Target Cost,Squared Loss,Cross Entropy Loss,Correct Predictions\n";
    eval_file << "Test " << n_epoch << "," << test_cost_target << "," << test_cost_v << "," << test_cost_p << "," << correct_policy_predictions << "\n";
    eval_file.close();
    return 0;
}
