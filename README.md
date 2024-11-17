# Simulation of Distributions and Statistical Sampling for Probabilistic Data Analysis and Visualization

This project is designed to generate, analyze, and visualize probabilistic data through the simulation of exponential and normal distributions, leveraging advanced statistical techniques and Python programming. By creating and exploring random samples of varying sizes, it provides insights into the properties of these distributions and their real-world applications through statistical calculations and graphical representations.

## Project Objectives

1. **Probabilistic Sampling Simulation**: Generate random samples from exponential and normal distributions using statistical and mathematical methods to represent natural and social processes.  
2. **Statistical Analysis**: Perform calculations of mean, kurtosis, skewness, and variance to examine the properties of each sample and evaluate its behavior relative to the theoretical distribution.  
3. **Graphical Visualization**: Create histograms overlaid with theoretical probability density curves to observe the central tendency and dispersion of the simulated data.  
4. **Results Exportation**: Store the generated data and calculated statistics in CSV files for external analysis and integration into other studies or predictive models.

## Motivation and Applications

Probabilistic data simulation is critical in fields such as statistics, finance, biology, and engineering, where analyzing data following known distributions is key to making predictions or assessing risks. Techniques used in this project have applications ranging from modeling the lifespan of components in engineering to estimating waiting times in queueing theory and forecasting rare events in financial risk analysis.

## Project Structure

### 1. Sample Generation and Statistical Calculations  

- Exponential samples of sizes \( n = 10, 20, 50 \) are generated using the `expon.rvs` function from `scipy.stats`, with the `scale` parameter adjusted to fit the expected event occurrence rate.  
- Key statistics calculated for each sample include:  
  - **Mean**: Average value of each sample.  
  - **Kurtosis and Skewness**: To analyze the shape and symmetry of the generated distribution.

### 2. Kurtosis and Skewness Analysis  

Kurtosis and skewness are used to identify significant deviations of the sample from the ideal distribution, providing insights into data dispersion and the propensity for extreme values.

### 3. Histogram Visualization  

To enable visual comparison between the generated samples and their theoretical distributions, histograms overlaid with theoretical probability density curves are created. These visualizations highlight how well the samples align with the expected distribution, providing valuable information on simulation accuracy and data trends.

### 4. CSV Results Export  

The generated data and corresponding statistics are exported to CSV files, offering an organized record that facilitates external analysis and enables integration into larger studies.

## Requirements  

To run this project, you need:  

- **Python 3.x**  
- **Libraries**:  
  - `pandas`  
  - `numpy`  
  - `seaborn`  
  - `matplotlib`  
  - `scipy`  

## Contributions  

This project is open to contributions. If you have suggestions or improvements for the simulation and analysis of probabilistic distributions, feel free to submit a pull request or open an issue.