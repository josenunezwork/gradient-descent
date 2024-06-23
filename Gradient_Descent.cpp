#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// Gradient Descent class for Linear Regression
class GradientDescent {
private:
    double learning_rate;    // Step size for parameter updates
    int max_iterations;      // Maximum number of iterations
    double tolerance;        // Convergence criterion

public:
    // Constructor with default hyperparameters
    GradientDescent(double lr = 0.01, int max_iter = 1000, double tol = 1e-6)
        : learning_rate(lr), max_iterations(max_iter), tolerance(tol) {}

    // Fit method to train the model
    std::pair<double, double> fit(const std::vector<double>& X, const std::vector<double>& y) {
        // Initialize model parameters
        double m = 0.0; // slope (weight)
        double b = 0.0; // y-intercept (bias)

        int n = X.size();
        
        // Gradient Descent iterations
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Initialize gradient accumulators
            double dm = 0.0; // Gradient for slope
            double db = 0.0; // Gradient for y-intercept

            // Compute gradients
            for (int i = 0; i < n; ++i) {
                double prediction = m * X[i] + b;  // Linear model: y = mx + b
                double error = prediction - y[i];  // Prediction error
                
                // Accumulate gradients
                dm += error * X[i];  // dL/dm = x * (y_pred - y_true)
                db += error;         // dL/db = (y_pred - y_true)
            }

            // Average gradients over all samples
            dm /= n;
            db /= n;

            // Update parameters using gradient descent
            double new_m = m - learning_rate * dm;
            double new_b = b - learning_rate * db;

            // Check for convergence
            if (std::abs(new_m - m) < tolerance && std::abs(new_b - b) < tolerance) {
                break;
            }

            // Update parameters
            m = new_m;
            b = new_b;
        }

        // Return final parameters
        return {m, b};
    }
};

// generate synthetic data
std::pair<std::vector<double>, std::vector<double>> generate_data(int n, double true_m, double true_b, double noise_std) {
    std::vector<double> X, y;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, noise_std);

    for (int i = 0; i < n; ++i) {
        double x = i / static_cast<double>(n);
        X.push_back(x);
        y.push_back(true_m * x + true_b + distribution(generator));
    }

    return {X, y};
}

int main() {
    // Hyperparameters for the model
    double learning_rate = 0.1;
    int max_iterations = 1000;
    double tolerance = 1e-6;

    // Generate synthetic data
    int n_samples = 100;
    double true_slope = 2.0;
    double true_intercept = 1.0;
    double noise_std = 0.1;
    auto [X, y] = generate_data(n_samples, true_slope, true_intercept, noise_std);

    // train the model
    GradientDescent gd(learning_rate, max_iterations, tolerance);
    auto [estimated_m, estimated_b] = gd.fit(X, y);


    std::cout << "True equation: y = " << true_slope << "x + " << true_intercept << std::endl;
    std::cout << "Estimated equation: y = " << estimated_m << "x + " << estimated_b << std::endl;

    return 0;
}
