import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Union, Optional, Tuple, Type, Callable
import warnings

class FST:
"""
Fixed Set Training /Fundamental Split Technique (FST): A robust regression approach that iteratively identifies
and separates problematic data points from a dataset to improve model performance.

The algorithm works by running multiple random train-test splits and tracking which  
data points consistently lead to poor model performance when included in test sets.  
These points are then "fixed" into the training set for future iterations.  
"""  
  
def __init__(  
    self,  
    regressor=None,  
    bad_r2_threshold: float = -0.01,  
    outer_split_bad_frac: float = 0.30,  
    max_fixed_frac: float = 0.80,  
    n_inner: int = 100,  
    random_state: Optional[int] = None,  
    verbose: bool = True  
):  
    """  
    Initialize the FST instance with configuration parameters.  
      
    Parameters:  
    -----------  
    regressor : sklearn regressor or None, default=None  
        Regression model to use. If None, LinearRegression will be used.  
        Must implement fit(X, y) and predict(X) methods.  
    bad_r2_threshold : float, default=-0.01  
        R² threshold below which a model run is considered "bad" for the test sample.  
    outer_split_bad_frac : float, default=0.30  
        Fraction of worst non-fixed points (by bad ratio) to fix per outer iteration.  
    max_fixed_frac : float, default=0.80  
        Stop if fixed points reach this fraction of total data.  
    n_inner : int, default=100  
        Number of inner random splits per outer iteration.  
    random_state : int, optional  
        Random seed for reproducibility.  
    verbose : bool, default=True  
        Whether to print progress information.  
    """  
    self.regressor = regressor if regressor is not None else LinearRegression()  
    self.bad_r2_threshold = bad_r2_threshold  
    self.outer_split_bad_frac = outer_split_bad_frac  
    self.max_fixed_frac = max_fixed_frac  
    self.n_inner = n_inner  
    self.random_state = random_state  
    self.verbose = verbose  
      
    # Validate parameters  
    if not (0 <= self.outer_split_bad_frac <= 1):  
        raise ValueError("outer_split_bad_frac must be between 0 and 1")  
    if not (0 < self.max_fixed_frac <= 1):  
        raise ValueError("max_fixed_frac must be between 0 and 1")  
    if self.n_inner <= 0:  
        raise ValueError("n_inner must be greater than 0")  
      
    # Verify that regressor has fit and predict methods  
    if not (hasattr(self.regressor, 'fit') and hasattr(self.regressor, 'predict')):  
        raise ValueError("Regressor must implement fit(X, y) and predict(X) methods")  
      
    # Initialize attributes that will be set during fitting  
    self.model = None  
    self.fixed_points: Set[int] = set()  
    self.non_fixed_points: Set[int] = set()  
    self.total_points: Set[int] = set()  
    self.all_outer_r2: List[float] = []  
    self.feature_names: List[str] = []  
    self._is_fitted = False  
      
def _log(self, message: str) -> None:  
    """Print message if verbose mode is enabled."""  
    if self.verbose:  
        print(message)  
          
def _create_random_state(self) -> int:  
    """Generate a random state if not provided."""  
    if self.random_state is not None:  
        return self.random_state  
    return np.random.randint(0, 10000)  
  
def prepare_data(  
    self,   
    df: pd.DataFrame,  
    target_col: str,  
    feature_cols: Optional[List[str]] = None,  
    categorical_cols: Optional[List[str]] = None,  
    date_col: Optional[str] = None  
) -> Tuple[pd.DataFrame, pd.Series]:  
    """  
    Prepare data for modeling by handling missing values, encoding categorical features,  
    and extracting features from date columns if specified.  
      
    Parameters:  
    -----------  
    df : pd.DataFrame  
        Input DataFrame containing the data.  
    target_col : str  
        Name of the target column to predict.  
    feature_cols : list of str, optional  
        List of feature column names to use. If None, all columns except target will be used.  
    categorical_cols : list of str, optional  
        List of categorical columns to one-hot encode.  
    date_col : str, optional  
        Name of date column to extract month and day of week features from.  
          
    Returns:  
    --------  
    X : pd.DataFrame  
        Feature matrix  
    y : pd.Series  
        Target variable  
    """  
    # Create a copy to avoid modifying the original dataframe  
    df_model = df.copy()  
      
    # Drop rows with missing values in essential columns  
    essential_cols = [target_col]  
    if feature_cols:  
        essential_cols.extend(feature_cols)  
    if categorical_cols:  
        essential_cols.extend(categorical_cols)  
    if date_col:  
        essential_cols.append(date_col)  
          
    df_model = df_model.dropna(subset=essential_cols)  
      
    # Extract features from date column if specified  
    if date_col:  
        df_model[f'{date_col}_Month'] = df_model[date_col].dt.month  
        df_model[f'{date_col}_DayOfWeek'] = df_model[date_col].dt.dayofweek  
      
    # One-hot encode categorical features if specified  
    if categorical_cols:  
        df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)  
      
    # Define features (X) and target (y)  
    if feature_cols is None:  
        # Use all columns except target as features  
        feature_cols = [col for col in df_model.columns if col != target_col]  
          
        # Include derived date features if any  
        if date_col:  
            date_derived = [f'{date_col}_Month', f'{date_col}_DayOfWeek']  
            feature_cols.extend([col for col in date_derived if col not in feature_cols])  
              
        # Include one-hot encoded features if any  
        if categorical_cols:  
            for cat_col in categorical_cols:  
                encoded_cols = [col for col in df_model.columns if col.startswith(f"{cat_col}_")]  
                feature_cols.extend([col for col in encoded_cols if col not in feature_cols])  
      
    # Store feature names for later use  
    self.feature_names = feature_cols  
      
    X = df_model[feature_cols]  
    y = df_model[target_col]  
      
    return X, y  
      
def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FST':  
    """  
    Fit the FST model by iteratively identifying and fixing problematic data points.  
      
    Parameters:  
    -----------  
    X : pd.DataFrame  
        Feature matrix.  
    y : pd.Series  
        Target variable.  
          
    Returns:  
    --------  
    self : FST  
        The fitted model instance.  
    """  
    if not isinstance(X, pd.DataFrame):  
        raise TypeError("X must be a pandas DataFrame")  
          
    if not isinstance(y, pd.Series):  
        raise TypeError("y must be a pandas Series")  
          
    if len(X) != len(y):  
        raise ValueError("X and y must have the same number of samples")  
          
    # Reset attributes  
    self.fixed_points = set()  
    self.all_outer_r2 = []  
      
    # Initialize data points  
    self.total_points = set(X.index)  
    self.non_fixed_points = self.total_points.copy()  
      
    # Store feature names  
    self.feature_names = X.columns.tolist()  
      
    outer_iter = 0  
      
    while True:  
        outer_iter += 1  
        self._log(f"\n=== Outer Iteration {outer_iter} ===")  
          
        # Prepare counters to track performance on non-fixed points  
        test_appearances = defaultdict(int)  
        bad_counts = defaultdict(int)  
        r2_list = []  # store R² of each inner split  
          
        # Work with non-fixed points as a list  
        non_fixed_list = list(self.non_fixed_points)  
        if len(non_fixed_list) == 0:  
            self._log("No non-fixed points remaining.")  
            break  
              
        # Run n_inner random splits on non-fixed points with a 70:30 split  
        for run in range(self.n_inner):  
            # Create a 70:30 split using only non-fixed indices  
            idx_train_nf, idx_test_nf = train_test_split(  
                non_fixed_list,   
                test_size=0.30,  
                random_state=np.random.randint(0, 10000)  
            )  
              
            # Overall training: fixed points U train from non-fixed  
            current_train_indices = list(self.fixed_points) + idx_train_nf  
            current_test_indices = idx_test_nf  # Test only from non-fixed  
              
            X_train = X.loc[current_train_indices]  
            y_train = y.loc[current_train_indices]  
            X_test = X.loc[current_test_indices]  
            y_test = y.loc[current_test_indices]  
              
            # Clone the regressor to ensure a fresh instance for each inner run  
            inner_model = self._clone_regressor()  
            inner_model.fit(X_train, y_train)  
            y_pred = inner_model.predict(X_test)  
            r2_val = r2_score(y_test, y_pred)  
            r2_list.append(r2_val)  
              
            # Track appearance and bad count for each test sample  
            for idx in current_test_indices:  
                test_appearances[idx] += 1  
                if r2_val <= self.bad_r2_threshold:  
                    bad_counts[idx] += 1  
                      
        avg_r2 = np.mean(r2_list)  
        self.all_outer_r2.append(avg_r2)  
        self._log(f"Average R² in outer iteration {outer_iter}: {avg_r2:.4f}")  
          
        # Update non-fixed points: Compute bad ratio and fix worst ones  
        bad_ratio = {}  
        for idx in self.non_fixed_points:  
            if test_appearances[idx] > 0:  
                bad_ratio[idx] = bad_counts[idx] / test_appearances[idx]  
                  
        if len(bad_ratio) == 0:  
            self._log("No test appearances among non-fixed points. Ending iteration.")  
            break  
              
        # Sort non-fixed points by descending bad ratio  
        sorted_bad = sorted(bad_ratio.items(), key=lambda x: x[1], reverse=True)  
        n_to_fix = int(len(sorted_bad) * self.outer_split_bad_frac)  
        new_fixed = set([idx for idx, ratio in sorted_bad[:n_to_fix]])  
        self._log(f"New fixed points in this iteration: {len(new_fixed)}")  
          
        if len(new_fixed) == 0:  
            self._log("No new bad points found. Stopping iterations.")  
            break  
              
        # Update fixed and non-fixed sets  
        self.fixed_points.update(new_fixed)  
        self.non_fixed_points = self.non_fixed_points - new_fixed  
          
        # Check stopping condition: if fixed set reaches max_fixed_frac of total data, then stop  
        if len(self.fixed_points) / len(self.total_points) >= self.max_fixed_frac:  
            self._log(f"Fixed points reached {self.max_fixed_frac*100}% of total data. Stopping iterations.")  
            break  
              
    # Train final model using fixed points  
    self._log("\n=== Training Final Model ===")  
    self._log(f"Total data points: {len(self.total_points)}")  
    self._log(f"Fixed points (final training set): {len(self.fixed_points)}")  
    self._log(f"Non-fixed points (final test set): {len(self.non_fixed_points)}")  
      
    if len(self.fixed_points) > 0:  
        X_train_final = X.loc[list(self.fixed_points)]  
        y_train_final = y.loc[list(self.fixed_points)]  
        self.model = self._clone_regressor()  
        self.model.fit(X_train_final, y_train_final)  
        self._is_fitted = True  
    else:  
        warnings.warn("No fixed points were identified. Model not fitted.")  
          
    return self  
  
def _clone_regressor(self):  
    """Create a fresh clone of the regressor with the same parameters."""  
    try:  
        from sklearn.base import clone  
        return clone(self.regressor)  
    except:  
        # If sklearn's clone doesn't work, try a simple copy  
        import copy  
        return copy.deepcopy(self.regressor)  
  
def predict(self, X: pd.DataFrame) -> np.ndarray:  
    """  
    Make predictions using the fitted model.  
      
    Parameters:  
    -----------  
    X : pd.DataFrame  
        Feature matrix for prediction.  
          
    Returns:  
    --------  
    np.ndarray  
        Predicted values.  
    """  
    if not self._is_fitted:  
        raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")  
          
    return self.model.predict(X)  
  
def score(self, X: pd.DataFrame, y: pd.Series) -> float:  
    """  
    Calculate the R² score of the model on the given data.  
      
    Parameters:  
    -----------  
    X : pd.DataFrame  
        Feature matrix.  
    y : pd.Series  
        Target variable.  
          
    Returns:  
    --------  
    float  
        R² score.  
    """  
    if not self._is_fitted:  
        raise ValueError("Model is not fitted yet. Call 'fit' before 'score'.")  
          
    y_pred = self.predict(X)  
    return r2_score(y, y_pred)  
  
def get_robust_indices(self) -> Dict[str, Set[int]]:  
    """  
    Get the indices of fixed (training) and non-fixed (test) data points.  
      
    Returns:  
    --------  
    Dict[str, Set[int]]  
        Dictionary with keys 'fixed' and 'non_fixed' containing the respective indices.  
    """  
    return {  
        'fixed': self.fixed_points,  
        'non_fixed': self.non_fixed_points  
    }  
  
def evaluate_partition_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:  
    """  
    Evaluate the performance of the partitioning approach compared to a random split.  
      
    Parameters:  
    -----------  
    X : pd.DataFrame  
        Feature matrix.  
    y : pd.Series  
        Target variable.  
          
    Returns:  
    --------  
    Dict[str, float]  
        Dictionary containing performance metrics.  
    """  
    if not self._is_fitted:  
        raise ValueError("Model is not fitted yet. Call 'fit' before evaluating.")  
          
    results = {}  
      
    # Evaluate on non-fixed points (test set)  
    if len(self.non_fixed_points) > 0:  
        X_test = X.loc[list(self.non_fixed_points)]  
        y_test = y.loc[list(self.non_fixed_points)]  
        y_pred = self.predict(X_test)  
        partition_r2 = r2_score(y_test, y_pred)  
        results['robust_partition_r2'] = partition_r2  
        self._log(f"R² on robust partition (non-fixed points): {partition_r2:.4f}")  
    else:  
        results['robust_partition_r2'] = None  
        self._log("No non-fixed points available for evaluation.")  
          
    # Compare with random split  
    train_ratio = len(self.fixed_points) / len(self.total_points)  
    test_ratio = 1 - train_ratio  
      
    if test_ratio <= 0:  
        test_ratio = 0.2  # Default to 80:20 split if all points are fixed  
          
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(  
        X, y, test_size=test_ratio, random_state=42  
    )  
    rand_model = self._clone_regressor()  
    rand_model.fit(X_train_rand, y_train_rand)  
    y_pred_rand = rand_model.predict(X_test_rand)  
    rand_r2 = r2_score(y_test_rand, y_pred_rand)  
    results['random_split_r2'] = rand_r2  
    self._log(f"R² using random {int((1-test_ratio)*100)}:{int(test_ratio*100)} split: {rand_r2:.4f}")  
      
    # Calculate improvement  
    if results['robust_partition_r2'] is not None:  
        results['improvement'] = results['robust_partition_r2'] - results['random_split_r2']  
        self._log(f"Improvement using robust partition: {results['improvement']:.4f}")  
      
    return results  
  
def plot_r2_progression(self, figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:  
    """  
    Plot the progression of average R² scores over outer iterations.  
      
    Parameters:  
    -----------  
    figsize : tuple of int, default=(10, 5)  
        Figure size (width, height) in inches.  
          
    Returns:  
    --------  
    matplotlib.figure.Figure  
        The figure object containing the plot.  
    """  
    if not self.all_outer_r2:  
        warnings.warn("No R² progression data available. Run fit() first.")  
        return None  
          
    fig = plt.figure(figsize=figsize)  
    plt.plot(self.all_outer_r2, marker='o')  
    plt.xlabel("Outer Iteration")  
    plt.ylabel("Average R² Score")  
    plt.title(f"Average R² Score over Outer Iterations ({type(self.regressor).__name__})")  
    plt.grid(True)  
    return fig  
  
def get_feature_importances(self) -> Optional[pd.Series]:  
    """  
    Get the feature importances from the regression model if available.  
    For linear models, returns coefficients.  
    For tree-based models, returns feature_importances_.  
      
    Returns:  
    --------  
    pd.Series or None  
        Feature importances as a pandas Series with feature names as index.  
        Returns None if the model doesn't support feature importances.  
    """  
    if not self._is_fitted:  
        raise ValueError("Model is not fitted yet. Call 'fit' before getting feature importances.")  
          
    if hasattr(self.model, 'coef_'):  
        importances = pd.Series(self.model.coef_, index=self.feature_names)  
        return importances.sort_values(ascending=False)  
    elif hasattr(self.model, 'feature_importances_'):  
        importances = pd.Series(self.model.feature_importances_, index=self.feature_names)  
        return importances.sort_values(ascending=False)  
    else:  
        warnings.warn(f"Model of type {type(self.model).__name__} doesn't provide feature importances.")  
        return None
