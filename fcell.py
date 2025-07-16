import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
try:
    from xgboost import XGBRegressor
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("XGBoost not installed. Install with: pip install xgboost")
import warnings
warnings.filterwarnings('ignore')

class FuelCellMLPipeline:
    """
    Modular Machine Learning Pipeline for Fuel Cell Voltage Prediction
    
    This pipeline implements multiple regression models (SVR, Random Forest, ANN)
    with automated hyperparameter optimization and comprehensive evaluation.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.feature_importance = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models with their respective hyperparameter grids."""
        
        # Support Vector Regression
        self.models['SVR'] = {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
        
        # Random Forest
        self.models['RandomForest'] = {
            'model': RandomForestRegressor(random_state=self.random_state),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        }
        
        # Artificial Neural Network (Multi-layer Perceptron)
        self.models['ANN'] = {
            'model': MLPRegressor(random_state=self.random_state, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'solver': ['adam', 'lbfgs']
            }
        }
        
        # XGBoost Regressor (if available)
        if xgboost_available:
            self.models['XGBoost'] = {
                'model': XGBRegressor(random_state=self.random_state, verbosity=0),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2]
                }
            }
    
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic fuel cell data for demonstration.
        In practice, you would load your actual fuel cell dataset here.
        """
        np.random.seed(self.random_state)
        
        # Simulate fuel cell operating parameters
        temperature = np.random.normal(70, 10, n_samples)  # °C
        pressure = np.random.normal(1.5, 0.3, n_samples)   # atm
        humidity = np.random.uniform(20, 80, n_samples)     # %
        current_density = np.random.uniform(0.1, 1.5, n_samples)  # A/cm²
        flow_rate = np.random.uniform(50, 200, n_samples)   # ml/min
        
        # Create realistic voltage based on fuel cell physics
        # Simplified model: V = V_oc - losses
        V_oc = 1.2  # Open circuit voltage
        
        # Various losses
        activation_loss = 0.1 * np.log(current_density + 0.01)
        ohmic_loss = 0.2 * current_density
        concentration_loss = 0.05 * (current_density ** 2)
        
        # Temperature and pressure effects
        temp_effect = (temperature - 25) * 0.001
        pressure_effect = np.log(pressure) * 0.05
        humidity_effect = (humidity - 50) * 0.0005
        flow_effect = (flow_rate - 100) * 0.0001
        
        # Calculate voltage with some noise
        voltage = (V_oc - activation_loss - ohmic_loss - concentration_loss + 
                  temp_effect + pressure_effect + humidity_effect + flow_effect +
                  np.random.normal(0, 0.02, n_samples))
        
        # Ensure realistic voltage range
        voltage = np.clip(voltage, 0.3, 1.2)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Temperature': temperature,
            'Pressure': pressure,
            'Humidity': humidity,
            'Current_Density': current_density,
            'Flow_Rate': flow_rate,
            'Voltage': voltage
        })
        
        return data
    
    def load_data(self, data=None, target_column='Voltage'):
        """
        Load and prepare data for training.
        
        Args:
            data: DataFrame with fuel cell data (if None, generates synthetic data)
            target_column: Name of the target variable column
        """
        if data is None:
            print("Generating synthetic fuel cell data...")
            data = self.generate_synthetic_data()
        
        self.data = data
        self.target_column = target_column
        
        # Separate features and target
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]
        
        print(f"Data loaded: {len(data)} samples, {len(self.X.columns)} features")
        print(f"Features: {list(self.X.columns)}")
        
        return self.X, self.y
    
    def preprocess_data(self, test_size=0.2):
        """
        Preprocess data: split, scale, and prepare for training.
        
        Args:
            test_size: Proportion of data to use for testing
        """
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data split: {len(self.X_train)} training, {len(self.X_test)} testing samples")
        
    def train_models(self, cv_folds=5, n_jobs=-1, verbose=True):
        """
        Train all models with hyperparameter optimization using GridSearchCV.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all processors)
            verbose: Whether to print progress
        """
        print("Training models with hyperparameter optimization...")
        
        for model_name, model_info in self.models.items():
            if verbose:
                print(f"\nTraining {model_name}...")
            
            # Prepare data based on model requirements
            if model_name in ['SVR', 'ANN']:
                X_train = self.X_train_scaled
            else:
                X_train = self.X_train
            
            # Grid Search with Cross-Validation
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                cv=cv_folds,
                scoring='r2',
                n_jobs=n_jobs,
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train, self.y_train)
            
            # Store best model
            self.best_models[model_name] = grid_search.best_estimator_
            
            if verbose:
                print(f"Best {model_name} score (CV): {grid_search.best_score_:.4f}")
                print(f"Best {model_name} params: {grid_search.best_params_}")
    
    def evaluate_models(self):
        """
        Evaluate all trained models on test data and store comprehensive metrics.
        """
        print("\nEvaluating models on test data...")
        
        for model_name, model in self.best_models.items():
            # Prepare test data
            if model_name in ['SVR', 'ANN']:
                X_test = self.X_test_scaled
            else:
                X_test = self.X_test
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            
            # Store results
            self.results[model_name] = {
                'R²': r2,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'predictions': y_pred
            }
            
            print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    def analyze_feature_importance(self):
        """
        Perform feature importance analysis for all models.
        """
        print("\nAnalyzing feature importance...")
        
        for model_name, model in self.best_models.items():
            if model_name in ['SVR', 'ANN']:
                X_test = self.X_test_scaled
            else:
                X_test = self.X_test
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models have built-in feature importance
                importance = model.feature_importances_
                method = 'Built-in'
            else:
                # Use permutation importance for other models
                perm_importance = permutation_importance(
                    model, X_test, self.y_test, 
                    n_repeats=10, random_state=self.random_state
                )
                importance = perm_importance.importances_mean
                method = 'Permutation'
            
            # Store feature importance
            self.feature_importance[model_name] = {
                'importance': importance,
                'features': self.X.columns.tolist(),
                'method': method
            }
    
    def plot_results(self, figsize=(15, 12)):
        """
        Create comprehensive visualization of results.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Fuel Cell Voltage Prediction - Model Performance Analysis', fontsize=16)
        
        # 1. Model Performance Comparison (R²)
        models = list(self.results.keys())
        r2_scores = [self.results[model]['R²'] for model in models]
        
        # Use more colors for more models
        colors = ['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'][:len(models)]
        
        axes[0, 0].bar(models, r2_scores, color=colors)
        axes[0, 0].set_title('Model Performance (R² Score)')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, score in enumerate(r2_scores):
            axes[0, 0].text(i, score + 0.01, f'{score:.3f}', ha='center')
        
        # 2. RMSE Comparison
        rmse_scores = [self.results[model]['RMSE'] for model in models]
        axes[0, 1].bar(models, rmse_scores, color=colors)
        axes[0, 1].set_title('Root Mean Square Error')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, score in enumerate(rmse_scores):
            axes[0, 1].text(i, score + max(rmse_scores)*0.01, f'{score:.3f}', ha='center')
        
        # 3. Predictions vs Actual (Best model)
        best_model = max(models, key=lambda x: self.results[x]['R²'])
        y_pred_best = self.results[best_model]['predictions']
        
        axes[0, 2].scatter(self.y_test, y_pred_best, alpha=0.6, color='blue')
        axes[0, 2].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 2].set_title(f'Predictions vs Actual ({best_model})')
        axes[0, 2].set_xlabel('Actual Voltage')
        axes[0, 2].set_ylabel('Predicted Voltage')
        
        # 4. Feature Importance for Random Forest
        if 'RandomForest' in self.feature_importance:
            rf_importance = self.feature_importance['RandomForest']
            feature_names = rf_importance['features']
            importance_values = rf_importance['importance']
            
            sorted_idx = np.argsort(importance_values)
            axes[1, 0].barh(range(len(feature_names)), 
                           [importance_values[i] for i in sorted_idx])
            axes[1, 0].set_yticks(range(len(feature_names)))
            axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[1, 0].set_title('Feature Importance (Random Forest)')
            axes[1, 0].set_xlabel('Importance')
        
        # 5. Residuals plot
        residuals = self.y_test - y_pred_best
        axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title(f'Residuals Plot ({best_model})')
        axes[1, 1].set_xlabel('Predicted Voltage')
        axes[1, 1].set_ylabel('Residuals')
        
        # 6. Data distribution
        axes[1, 2].hist(self.y, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 2].set_title('Target Variable Distribution')
        axes[1, 2].set_xlabel('Voltage')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_report(self):
        """
        Generate a comprehensive summary report.
        """
        print("\n" + "="*80)
        print("FUEL CELL VOLTAGE PREDICTION - SUMMARY REPORT")
        print("="*80)
        
        print(f"\nDataset Information:")
        print(f"- Total samples: {len(self.data)}")
        print(f"- Features: {len(self.X.columns)}")
        print(f"- Target range: {self.y.min():.3f} - {self.y.max():.3f} V")
        
        print(f"\nModel Performance Summary:")
        print("-" * 60)
        print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 60)
        
        for model_name in self.results:
            r2 = self.results[model_name]['R²']
            rmse = self.results[model_name]['RMSE']
            mae = self.results[model_name]['MAE']
            print(f"{model_name:<15} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f}")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['R²'])
        best_r2 = self.results[best_model]['R²']
        
        print(f"\nBest Model: {best_model} (R² = {best_r2:.4f})")
        
        if best_r2 > 0.9:
            print("[SUCCESS] Target achieved: R² > 0.9")
        else:
            print(f"[WARNING] Target not achieved: R² = {best_r2:.4f} < 0.9")
        
        print(f"\nTop 3 Most Important Features (Random Forest):")
        if 'RandomForest' in self.feature_importance:
            rf_importance = self.feature_importance['RandomForest']
            importance_data = list(zip(rf_importance['features'], rf_importance['importance']))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_data[:3]):
                print(f"{i+1}. {feature}: {importance:.4f}")

# Example usage and demonstration
def main():
    """
    Demonstration of the Fuel Cell ML Pipeline
    """
    print("Fuel Cell Voltage Prediction Pipeline")
    print("=====================================")
    
    # Initialize pipeline
    pipeline = FuelCellMLPipeline(random_state=42)
    
    # Load data (synthetic for demonstration)
    X, y = pipeline.load_data()
    
    # Preprocess data
    pipeline.preprocess_data(test_size=0.2)
    
    # Train models with hyperparameter optimization
    pipeline.train_models(cv_folds=5, verbose=True)
    
    # Evaluate models
    pipeline.evaluate_models()
    
    # Analyze feature importance
    pipeline.analyze_feature_importance()
    
    # Generate visualizations
    pipeline.plot_results()
    
    # Print summary report
    pipeline.get_summary_report()

if __name__ == "__main__":
    main()