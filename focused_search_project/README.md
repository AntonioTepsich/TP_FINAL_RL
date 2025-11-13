# Focused Hyperparameter Search for Reinforcement Learning

This project is designed to conduct a focused hyperparameter search for a reinforcement learning model, building upon the results obtained from an initial random search. The goal is to fine-tune the model parameters to achieve better performance.

## Files in the Project

- **phase2_focused_search.yaml**: This configuration file specifies the settings for the focused hyperparameter search. It includes a refined search space for key parameters such as learning rate, hidden size, and entropy, based on insights gained from the initial phase.

- **README.md**: This documentation file provides instructions on how to run the focused search, interpret the results, and understand the purpose of the configuration file.

## Running the Focused Search

To execute the focused hyperparameter search, use the following command in your terminal:

```bash
python run_experiment.py --config config_template.yaml --search --search-config phase2_focused_search.yaml
```

## Interpreting Results

After running the focused search, you can analyze the results using:

```bash
python view_results.py --bucket ppo-flappy-bird --best 6
```

### Key Metrics to Consider

- **Best Score**: Look for the highest score achieved during the trials.
- **Final Mean Score**: Evaluate the stability of the model by comparing the final mean score to the best score.
- **Parameter Impact**: Assess how changes in learning rate, hidden size, and entropy affected performance.

## Next Steps

1. If improvements are observed, consider further refining the hyperparameters in subsequent phases.
2. If no significant improvements are found, the current baseline may already be optimal.
3. Adjust the search ranges if any trials resulted in crashes or instability.

This project aims to enhance the performance of reinforcement learning models through systematic hyperparameter optimization.