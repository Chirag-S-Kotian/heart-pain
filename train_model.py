import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Enhanced imports for state-of-the-art implementation
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           average_precision_score, matthews_corrcoef)
from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# XGBoost and CatBoost - state-of-the-art performers
import xgboost as xgb
import catboost as cb

# SHAP for explainable AI
import shap

# Optuna for advanced hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce optuna output

import pickle
import os
import joblib
from scipy.stats import randint, uniform

# Set random seeds for reproducibility
np.random.seed(42)

class EnhancedHeartDiseasePredictor:
    """
    Complete Heart Disease Prediction System with Explainable AI
    FIXED VERSION - XGBoost data type issue resolved
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_selector = None
        self.shap_explainer = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.raw_feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        self.engineered_feature_names = []
        
    def enhanced_data_loading(self):
        """Enhanced data loading with multiple high-quality datasets"""
        print("Loading and combining multiple heart disease datasets...")
        
        # Load Cleveland dataset (most reliable)
        cleveland_df = self.load_cleveland_data()
        
        # Load additional datasets for robustness
        framingham_df = self.load_framingham_data()
        statlog_df = self.load_statlog_data()
        
        # Combine datasets with source tracking
        all_dfs = [df for df in [cleveland_df, framingham_df, statlog_df] if not df.empty]
        if not all_dfs:
            print("No datasets loaded successfully, creating sample data...")
            return self.create_sample_data()
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"Combined dataset shape: {combined_df.shape}")
        print("\nDataset sources distribution:")
        print(combined_df['source'].value_counts())
        
        return combined_df
    
    def create_sample_data(self):
        """Create sample data if real datasets can't be loaded"""
        print("Creating sample heart disease data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic sample data
        data = {
            'age': np.random.normal(55, 12, n_samples).clip(25, 85).astype(int),
            'sex': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.45, 0.25, 0.20, 0.10]),
            'trestbps': np.random.normal(132, 18, n_samples).clip(90, 200).astype(int),
            'chol': np.random.normal(246, 52, n_samples).clip(150, 450).astype(int),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.50, 0.45, 0.05]),
            'thalach': np.random.normal(149, 23, n_samples).clip(70, 202).astype(int),
            'exang': np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),
            'oldpeak': np.random.exponential(1, n_samples).clip(0, 6.2).round(1),
            'slope': np.random.choice([0, 1, 2], n_samples, p=[0.20, 0.65, 0.15]),
            'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.55, 0.25, 0.15, 0.05]),
            'thal': np.random.choice([1, 2, 3], n_samples, p=[0.15, 0.70, 0.15]),
            'source': ['sample'] * n_samples
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target based on clinical risk factors
        risk_score = (
            (df['age'] > 55).astype(int) * 1.8 +
            df['sex'] * 1.2 +
            (df['cp'] > 1).astype(int) * 2.1 +
            (df['trestbps'] > 140).astype(int) * 1.3 +
            (df['chol'] > 240).astype(int) * 0.8 +
            df['fbs'] * 0.9 +
            df['exang'] * 1.7 +
            (df['oldpeak'] > 1).astype(int) * 1.4 +
            (df['slope'] == 0).astype(int) * 1.1 +
            (df['ca'] > 0).astype(int) * 1.6 +
            (df['thal'] == 3).astype(int) * 1.5
        )
        
        # Convert risk score to probability and then to binary target
        prob = 1 / (1 + np.exp(-(risk_score - 5.5)))
        df['target'] = np.random.binomial(1, prob, n_samples)
        
        print(f"Sample data created with {len(df)} records")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def load_cleveland_data(self):
        """Load and preprocess Cleveland dataset"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(url, names=columns, na_values='?')
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
            df['source'] = 'cleveland'
            print("✅ Cleveland dataset loaded successfully")
            return df
        except Exception as e:
            print(f"❌ Error loading Cleveland dataset: {e}")
            return pd.DataFrame()
    
    def load_framingham_data(self):
        """Load Framingham dataset with proper mapping"""
        try:
            url = "https://raw.githubusercontent.com/rashida048/Datasets/master/framingham.csv"
            df = pd.read_csv(url)
            
            # Map to Cleveland schema
            df = df.rename(columns={
                'sysBP': 'trestbps',
                'totChol': 'chol',
                'diabetes': 'fbs',
                'heartRate': 'thalach',
                'TenYearCHD': 'target'
            })
            
            # Add missing columns with reasonable defaults
            df['cp'] = 0
            df['restecg'] = 0
            df['exang'] = 0
            df['oldpeak'] = 0
            df['slope'] = 1
            df['ca'] = 0
            df['thal'] = 2
            df['source'] = 'framingham'
            
            selected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target', 'source']
            
            # Filter existing columns
            available_cols = [col for col in selected_cols if col in df.columns]
            df_filtered = df[available_cols]
            
            print("✅ Framingham dataset loaded successfully")
            return df_filtered
        except Exception as e:
            print(f"❌ Error loading Framingham dataset: {e}")
            return pd.DataFrame()
    
    def load_statlog_data(self):
        """Load Statlog heart disease dataset"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(url, names=columns, sep=' ', na_values='?')
            df['target'] = df['target'].apply(lambda x: 0 if x == 1 else 1)
            df['source'] = 'statlog'
            print("✅ Statlog dataset loaded successfully")
            return df
        except Exception as e:
            print(f"❌ Error loading Statlog dataset: {e}")
            return pd.DataFrame()
    
    def advanced_preprocessing(self, df):
        """Enhanced preprocessing with state-of-the-art techniques"""
        print("Applying advanced preprocessing...")
        
        original_shape = df.shape
        
        # Remove rows with missing target
        df = df.dropna(subset=['target'])
        
        # Advanced missing value imputation using KNN
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        # Check if we have enough data for KNN imputation
        if len(df) > 10:
            knn_imputer = KNNImputer(n_neighbors=min(5, len(df)//3))
            df[numerical_features] = knn_imputer.fit_transform(df[numerical_features])
        else:
            # Fallback to median imputation for small datasets
            df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
        
        # Fill remaining missing values strategically
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else 0)
        
        # Remove outliers using IQR method for key features (only if we have enough data)
        if len(df) > 50:
            for feature in ['trestbps', 'chol', 'thalach']:
                if feature in df.columns:
                    Q1 = df[feature].quantile(0.25)
                    Q3 = df[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
        
        print(f"Dataset shape: {original_shape} → {df.shape}")
        return df
    
    def advanced_feature_engineering(self, df):
        """State-of-the-art feature engineering - FIXED for XGBoost compatibility"""
        print("Creating advanced engineered features...")
        
        df = df.copy()
        
        # Ensure all basic features exist
        for feature in self.raw_feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Convert all basic features to numeric to avoid any object dtype issues
        for feature in self.raw_feature_names:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # Age-based risk stratification (convert to numeric features directly)
        df['age_very_low'] = (df['age'] <= 40).astype(int)
        df['age_low'] = ((df['age'] > 40) & (df['age'] <= 50)).astype(int)
        df['age_moderate'] = ((df['age'] > 50) & (df['age'] <= 60)).astype(int)
        df['age_high'] = ((df['age'] > 60) & (df['age'] <= 70)).astype(int)
        df['age_very_high'] = (df['age'] > 70).astype(int)
        
        # Blood pressure categories (numeric features)
        df['bp_normal'] = (df['trestbps'] <= 120).astype(int)
        df['bp_elevated'] = ((df['trestbps'] > 120) & (df['trestbps'] <= 130)).astype(int)
        df['bp_stage1_htn'] = ((df['trestbps'] > 130) & (df['trestbps'] <= 140)).astype(int)
        df['bp_stage2_htn'] = ((df['trestbps'] > 140) & (df['trestbps'] <= 180)).astype(int)
        df['bp_crisis'] = (df['trestbps'] > 180).astype(int)
        
        # Cholesterol risk levels (numeric features)
        df['chol_desirable'] = (df['chol'] <= 200).astype(int)
        df['chol_borderline'] = ((df['chol'] > 200) & (df['chol'] <= 240)).astype(int)
        df['chol_high'] = ((df['chol'] > 240) & (df['chol'] <= 280)).astype(int)
        df['chol_very_high'] = (df['chol'] > 280).astype(int)
        
        # Heart rate calculations
        df['max_hr_predicted'] = 220 - df['age']
        df['hr_reserve'] = df['max_hr_predicted'] - df['thalach']
        df['hr_response_ratio'] = df['thalach'] / (df['max_hr_predicted'] + 1e-8)
        
        # Advanced cardiovascular risk scores
        df['framingham_risk_score'] = (
            (df['age'] - 40) * 0.1 +
            df['sex'] * 2.5 +
            (df['trestbps'] - 120) * 0.02 +
            (df['chol'] - 200) * 0.01 +
            df['fbs'] * 1.5 +
            df['exang'] * 2.0
        )
        
        # Interaction features (based on clinical knowledge)
        df['age_chol_interaction'] = df['age'] * df['chol'] / 1000
        df['age_bp_interaction'] = df['age'] * df['trestbps'] / 1000
        df['sex_age_interaction'] = df['sex'] * df['age']
        
        # Exercise capacity indicators
        df['exercise_tolerance'] = (df['thalach'] > 150).astype(int)
        df['poor_exercise_response'] = ((df['exang'] == 1) | (df['oldpeak'] > 2)).astype(int)
        
        # Ensure ALL columns are numeric (critical for XGBoost)
        for col in df.columns:
            if col not in ['source']:  # Skip non-numeric source column
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        
        print(f"Features after engineering: {df.shape[1]} features")
        print(f"All feature dtypes are numeric: {all(df.drop('source', axis=1, errors='ignore').dtypes != 'object')}")
        
        return df
    
    def create_feature_pipeline_for_prediction(self, raw_patient_data):
        """Apply identical feature engineering for prediction - FIXED VERSION"""
        
        # Create DataFrame from raw data
        df = pd.DataFrame([raw_patient_data])
        
        # Ensure all basic features exist with default values
        for feature in self.raw_feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Apply the EXACT same feature engineering as training
        return self.advanced_feature_engineering(df)
    
    def shap_based_feature_selection(self, X, y, n_features=15):
        """Advanced feature selection using SHAP values"""
        print(f"Performing SHAP-based feature selection for top {n_features} features...")
        
        # Ensure all features are numeric
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Train a preliminary XGBoost model for SHAP analysis
        temp_model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        temp_model.fit(X_numeric, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(temp_model)
        shap_values = explainer.shap_values(X_numeric)
        
        # Get feature importance based on mean absolute SHAP values
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = feature_importance_df.head(n_features)['feature'].tolist()
        
        print("Top selected features by SHAP importance:")
        print(feature_importance_df.head(n_features))
        
        return selected_features, feature_importance_df
    
    def optimize_hyperparameters(self, X_train, y_train, model_type='xgboost', n_trials=15):
        """Advanced hyperparameter optimization using Optuna"""
        print(f"Optimizing {model_type} hyperparameters...")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 3),
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'verbosity': 0
                }
                model = xgb.XGBClassifier(**params)
            
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 300),
                    'depth': trial.suggest_int('depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.25),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 8),
                    'border_count': trial.suggest_int('border_count', 32, 128),
                    'random_state': 42,
                    'verbose': False
                }
                model = cb.CatBoostClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                      scoring='roc_auc', n_jobs=-1)
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"Best {model_type} ROC-AUC: {study.best_value:.4f}")
        
        return study.best_params
    
    def build_state_of_art_ensemble(self, X_train, y_train):
        """Build state-of-the-art ensemble model"""
        print("Building state-of-the-art ensemble model...")
        
        try:
            # Optimize XGBoost
            xgb_params = self.optimize_hyperparameters(X_train, y_train, 'xgboost', n_trials=10)
            xgb_model = xgb.XGBClassifier(**xgb_params)
        except:
            print("Using default XGBoost parameters")
            xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                        random_state=42, eval_metric='logloss', verbosity=0)
        
        try:
            # Optimize CatBoost  
            cb_params = self.optimize_hyperparameters(X_train, y_train, 'catboost', n_trials=10)
            cb_model = cb.CatBoostClassifier(**cb_params)
        except:
            print("Using default CatBoost parameters")
            cb_model = cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, 
                                           random_state=42, verbose=False)
        
        # Additional strong models
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=4,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=42
        )
        
        lr_model = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000,
            random_state=42
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('cb', cb_model),
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ],
            voting='soft'
        )
        
        # Store models
        self.models = {
            'xgboost': xgb_model,
            'catboost': cb_model,
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'logistic_regression': lr_model,
            'ensemble': ensemble
        }
        
        return ensemble
    
    def train_and_evaluate(self, df):
        """Complete training and evaluation pipeline"""
        print("Starting comprehensive training and evaluation...")
        
        # Preprocessing
        df = self.advanced_preprocessing(df)
        df = self.advanced_feature_engineering(df)
        
        # Prepare features and target
        X = df.drop(['target', 'source'], axis=1, errors='ignore')
        y = df['target']
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"All features are numeric: {all(X.dtypes != 'object')}")
        
        # Handle class imbalance with SMOTE
        if len(X) > 50 and len(np.unique(y)) > 1:
            print("Handling class imbalance with SMOTE...")
            min_samples = min(len(X[y==0]), len(X[y==1]))
            k_neighbors = min(5, min_samples - 1)
            
            if k_neighbors > 0:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_balanced, y_balanced = smote.fit_resample(X, y)
                print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
            else:
                X_balanced, y_balanced = X, y
                print("Skipping SMOTE due to insufficient samples")
        else:
            X_balanced, y_balanced = X, y
            print("Skipping SMOTE due to small dataset or single class")
        
        # Feature selection using SHAP
        n_features = min(15, len(X_balanced.columns))
        selected_features, feature_importance = self.shap_based_feature_selection(
            X_balanced, y_balanced, n_features=n_features
        )
        X_selected = X_balanced[selected_features]
        
        # Store feature names for prediction
        self.feature_names = selected_features
        self.engineered_feature_names = list(X_balanced.columns)
        
        print(f"Selected features: {self.feature_names}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_balanced, test_size=0.2, random_state=42, 
            stratify=y_balanced if len(np.unique(y_balanced)) > 1 else None
        )
        
        # Scale features
        self.scaler = RobustScaler()
        numerical_features = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if len(numerical_features) > 0:
            X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
            X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Build and train ensemble
        ensemble_model = self.build_state_of_art_ensemble(X_train_scaled, y_train)
        
        # Train all models
        print("Training all models...")
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                model.fit(X_train_scaled, y_train)
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # Calibrate the best model (ensemble)
        print("Calibrating ensemble model...")
        try:
            calibrated_model = CalibratedClassifierCV(ensemble_model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)
            self.best_model = calibrated_model
        except:
            print("Using uncalibrated ensemble model")
            self.best_model = ensemble_model
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)
        
        # SHAP analysis for interpretability
        self.shap_analysis(X_test_scaled)
        
        # Find optimal threshold
        self.find_optimal_threshold(X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'avg_precision': avg_precision,
                    'mcc': mcc
                }
                
                print(f"\n{name.upper()} Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"ROC-AUC: {roc_auc:.4f}")
                print(f"Average Precision: {avg_precision:.4f}")
                print(f"Matthews Correlation Coefficient: {mcc:.4f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        # Results summary
        if results:
            results_df = pd.DataFrame(results).T
            print("\n" + "="*60)
            print("MODEL COMPARISON SUMMARY")
            print("="*60)
            print(results_df.round(4))
            
            # Best model
            best_model_name = results_df['roc_auc'].idxmax()
            print(f"\nBest performing model: {best_model_name}")
            print(f"Best ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
            
            return results_df
        else:
            print("No model evaluation results available")
            return pd.DataFrame()
    
    def shap_analysis(self, X_test):
        """Comprehensive SHAP analysis for explainable AI"""
        print("\nPerforming SHAP analysis for explainable AI...")
        
        try:
            # Use XGBoost for SHAP analysis (most reliable)
            if 'xgboost' in self.models:
                xgb_model = self.models['xgboost']
                self.shap_explainer = shap.TreeExplainer(xgb_model)
                shap_values = self.shap_explainer.shap_values(X_test)
                
                # SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.title('SHAP Feature Importance Summary')
                plt.tight_layout()
                plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Feature importance summary
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features (SHAP Analysis):")
                print(importance_df.head(10))
                
                return shap_values, importance_df
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return None, None
    
    def find_optimal_threshold(self, X_test, y_test):
        """Find optimal classification threshold"""
        try:
            y_proba = self.best_model.predict_proba(X_test)[:, 1]
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            
            # Find optimal threshold (maximize F1 score)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            self.optimal_threshold = thresholds[optimal_idx]
            
            print(f"\nOptimal classification threshold: {self.optimal_threshold:.4f}")
            print(f"Optimal F1 score: {f1_scores[optimal_idx]:.4f}")
            
            return self.optimal_threshold
        except Exception as e:
            print(f"Error finding optimal threshold: {e}")
            self.optimal_threshold = 0.5
            return 0.5
    
    def create_visualizations(self, X_test, y_test):
        """Create comprehensive visualizations"""
        print("Creating advanced visualizations...")
        
        try:
            # Set up the figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Heart Disease Prediction Model Evaluation', fontsize=16, fontweight='bold')
            
            # ROC curves comparison
            ax1 = axes[0, 0]
            for name, model in self.models.items():
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    auc_score = roc_auc_score(y_test, y_proba)
                    ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                except:
                    continue
            
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6)
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curves Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Precision-Recall curves
            ax2 = axes[0, 1]
            for name, model in self.models.items():
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    ap_score = average_precision_score(y_test, y_proba)
                    ax2.plot(recall, precision, label=f'{name} (AP = {ap_score:.3f})')
                except:
                    continue
            
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curves')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Confusion matrix for best model
            ax3 = axes[1, 0]
            y_pred_optimal = (self.best_model.predict_proba(X_test)[:, 1] >= self.optimal_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred_optimal)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                       xticklabels=['Low Risk', 'High Risk'],
                       yticklabels=['Low Risk', 'High Risk'])
            ax3.set_title('Confusion Matrix (Optimal Threshold)')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            
            # Feature importance
            ax4 = axes[1, 1]
            if hasattr(self, 'models') and 'random_forest' in self.models:
                rf_model = self.models['random_forest']
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=True).tail(10)
                
                ax4.barh(range(len(feature_imp)), feature_imp['importance'])
                ax4.set_yticks(range(len(feature_imp)))
                ax4.set_yticklabels(feature_imp['feature'])
                ax4.set_xlabel('Importance')
                ax4.set_title('Top 10 Feature Importance (Random Forest)')
            
            plt.tight_layout()
            plt.savefig('comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Visualizations saved successfully!")
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def save_model(self, filepath='enhanced_heart_disease_model_2025.pkl'):
        """Save the complete model pipeline"""
        model_package = {
            'best_model': self.best_model,
            'all_models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'raw_feature_names': self.raw_feature_names,
            'engineered_feature_names': self.engineered_feature_names,
            'optimal_threshold': self.optimal_threshold,
            'shap_explainer': self.shap_explainer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Enhanced model saved to {filepath}")
    
    def load_model(self, filepath='enhanced_heart_disease_model_2025.pkl'):
        """Load a previously saved model"""
        try:
            with open(filepath, 'rb') as f:
                model_package = pickle.load(f)
            
            self.best_model = model_package['best_model']
            self.models = model_package['all_models']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.raw_feature_names = model_package.get('raw_feature_names', self.raw_feature_names)
            self.engineered_feature_names = model_package.get('engineered_feature_names', [])
            self.optimal_threshold = model_package['optimal_threshold']
            self.shap_explainer = model_package['shap_explainer']
            
            print(f"✅ Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_with_explanation(self, patient_data):
        """Make prediction with detailed explanation - FINAL FIXED VERSION"""
        
        try:
            print(f"\n{'='*60}")
            print(f"HEART DISEASE RISK PREDICTION")
            print(f"{'='*60}")
            print(f"Input: {patient_data}")
            
            # Step 1: Apply feature engineering to raw patient data
            engineered_df = self.create_feature_pipeline_for_prediction(patient_data)
            print(f"✅ Feature engineering applied: {engineered_df.shape[1]} features created")
            
            # Step 2: Select only numeric features (excluding 'source' if present)
            engineered_df_numeric = engineered_df.select_dtypes(include=[np.number])
            print(f"✅ Numeric features only: {engineered_df_numeric.shape[1]} features")
            
            # Step 3: Create final feature vector with all expected features
            final_df = pd.DataFrame(columns=self.feature_names)
            final_df.loc[0] = 0.0  # Initialize with float zeros
            
            # Fill in available features
            for col in engineered_df_numeric.columns:
                if col in self.feature_names:
                    final_df.loc[0, col] = float(engineered_df_numeric.loc[0, col])
            
            # Ensure all values are float type
            final_df = final_df.astype(float)
            
            print(f"✅ Feature vector prepared: {len(self.feature_names)} features")
            print(f"✅ All features are numeric: {all(final_df.dtypes != 'object')}")
            
            # Step 4: Apply scaling
            if self.scaler is not None:
                numerical_features = final_df.select_dtypes(include=[np.number]).columns
                if len(numerical_features) > 0:
                    final_df[numerical_features] = self.scaler.transform(final_df[numerical_features])
                    print(f"✅ Feature scaling applied to {len(numerical_features)} features")
            
            # Step 5: Make prediction
            print("Making prediction...")
            prediction_proba = self.best_model.predict_proba(final_df)[0, 1]
            prediction = (prediction_proba >= self.optimal_threshold).astype(int)
            
            # Step 6: Display results
            print(f"\n{'='*50}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*50}")
            risk_level = "🔴 HIGH RISK" if prediction else "🟢 LOW RISK"
            print(f"Risk Assessment: {risk_level}")
            print(f"Confidence Score: {prediction_proba:.4f}")
            print(f"Risk Threshold: {self.optimal_threshold:.4f}")
            
            # Step 7: SHAP explanation
            if self.shap_explainer:
                try:
                    shap_values = self.shap_explainer.shap_values(final_df)
                    
                    print(f"\n{'='*50}")
                    print("KEY RISK FACTORS")
                    print(f"{'='*50}")
                    
                    feature_contributions = list(zip(self.feature_names, shap_values[0]))
                    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for i, (feature, contribution) in enumerate(feature_contributions[:8]):
                        direction = "↑ INCREASES" if contribution > 0 else "↓ DECREASES"
                        impact = abs(contribution)
                        print(f"{i+1:2d}. {feature:25s} {direction:12s} risk by {impact:.4f}")
                        
                except Exception as e:
                    print(f"⚠️  SHAP explanation failed: {e}")
            
            # Step 8: Clinical interpretation
            self.provide_clinical_interpretation(patient_data, prediction, prediction_proba)
            
            return prediction, prediction_proba
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def provide_clinical_interpretation(self, patient_data, prediction, confidence):
        """Provide clinical interpretation of results"""
        print(f"\n{'='*50}")
        print("CLINICAL INTERPRETATION")
        print(f"{'='*50}")
        
        risk_factors = []
        protective_factors = []
        
        # Analyze major risk factors
        age = patient_data.get('age', 0)
        sex = patient_data.get('sex', 0)
        cp = patient_data.get('cp', 0)
        trestbps = patient_data.get('trestbps', 0)
        chol = patient_data.get('chol', 0)
        fbs = patient_data.get('fbs', 0)
        exang = patient_data.get('exang', 0)
        oldpeak = patient_data.get('oldpeak', 0)
        
        # Age factor
        if age > 65:
            risk_factors.append(f"Advanced age ({age} years)")
        elif age < 40:
            protective_factors.append(f"Young age ({age} years)")
        
        # Gender factor
        if sex == 1:
            risk_factors.append("Male gender")
        else:
            protective_factors.append("Female gender")
        
        # Chest pain
        if cp > 2:
            risk_factors.append("Atypical chest pain")
        elif cp == 0:
            protective_factors.append("No chest pain")
        
        # Blood pressure
        if trestbps > 140:
            risk_factors.append(f"High blood pressure ({trestbps} mmHg)")
        elif trestbps < 120:
            protective_factors.append(f"Normal blood pressure ({trestbps} mmHg)")
        
        # Cholesterol
        if chol > 240:
            risk_factors.append(f"High cholesterol ({chol} mg/dl)")
        elif chol < 200:
            protective_factors.append(f"Good cholesterol level ({chol} mg/dl)")
        
        # Other factors
        if fbs:
            risk_factors.append("Elevated fasting blood sugar")
        if exang:
            risk_factors.append("Exercise-induced chest pain")
        if oldpeak > 2:
            risk_factors.append("Significant ST depression")
        
        # Display interpretation
        if risk_factors:
            print("🔴 Risk Factors Present:")
            for i, factor in enumerate(risk_factors, 1):
                print(f"   {i}. {factor}")
        
        if protective_factors:
            print("🟢 Protective Factors:")
            for i, factor in enumerate(protective_factors, 1):
                print(f"   {i}. {factor}")
        
        # Recommendations
        print(f"\n{'='*50}")
        print("RECOMMENDATIONS")
        print(f"{'='*50}")
        
        if prediction:
            print("🏥 HIGH RISK - Immediate medical consultation recommended")
            print("   • Schedule appointment with cardiologist")
            print("   • Consider stress testing and advanced cardiac imaging")
            print("   • Implement aggressive risk factor modification")
        else:
            print("✅ LOW RISK - Continue preventive care")
            print("   • Maintain regular check-ups")
            print("   • Continue healthy lifestyle practices")
            print("   • Monitor any risk factors present")
    
    def batch_predict(self, patient_list):
        """Predict for multiple patients"""
        results = []
        
        for i, patient_data in enumerate(patient_list):
            print(f"\n{'='*80}")
            print(f"PATIENT {i+1} ANALYSIS")
            print(f"{'='*80}")
            
            prediction, confidence = self.predict_with_explanation(patient_data)
            
            results.append({
                'patient_id': i+1,
                'prediction': 'High Risk' if prediction else 'Low Risk',
                'confidence': confidence,
                'data': patient_data
            })
        
        return results
    
    def simple_predict(self, patient_data):
        """Simple prediction without detailed explanation"""
        try:
            # Apply feature engineering
            engineered_df = self.create_feature_pipeline_for_prediction(patient_data)
            
            # Select numeric features only
            engineered_df_numeric = engineered_df.select_dtypes(include=[np.number])
            
            # Prepare final feature vector
            final_df = pd.DataFrame(columns=self.feature_names)
            final_df.loc[0] = 0.0
            
            for col in engineered_df_numeric.columns:
                if col in self.feature_names:
                    final_df.loc[0, col] = float(engineered_df_numeric.loc[0, col])
            
            # Ensure all values are float
            final_df = final_df.astype(float)
            
            # Apply scaling
            if self.scaler is not None:
                numerical_features = final_df.select_dtypes(include=[np.number]).columns
                if len(numerical_features) > 0:
                    final_df[numerical_features] = self.scaler.transform(final_df[numerical_features])
            
            # Make prediction
            prediction_proba = self.best_model.predict_proba(final_df)[0, 1]
            prediction = (prediction_proba >= self.optimal_threshold).astype(int)
            
            return prediction, prediction_proba
            
        except Exception as e:
            print(f"Simple prediction failed: {e}")
            return None, None

# Main execution function
def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED HEART DISEASE PREDICTION SYSTEM - 2025")
    print("Complete State-of-the-Art System with Explainable AI - FIXED")
    print("="*80)
    
    # Initialize predictor
    predictor = EnhancedHeartDiseasePredictor()
    
    # Load and process data
    df = predictor.enhanced_data_loading()
    
    if df.empty:
        print("❌ No data available for training")
        return None
    
    # Train and evaluate
    try:
        X_test, y_test = predictor.train_and_evaluate(df)
        
        # Create visualizations
        predictor.create_visualizations(X_test, y_test)
        
        # Save model
        predictor.save_model()
        
        print("\n" + "="*80)
        print("✅ SYSTEM COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features:")
        print("✅ State-of-the-art ensemble models (XGBoost + CatBoost)")
        print("✅ Advanced feature engineering with clinical knowledge")
        print("✅ SHAP-based feature selection and explainability")
        print("✅ Optuna hyperparameter optimization")
        print("✅ Class imbalance handling with SMOTE")
        print("✅ Comprehensive evaluation metrics")
        print("✅ Clinical-grade visualizations")
        print("✅ FIXED feature engineering pipeline (XGBoost compatible)")
        print("✅ Clinical interpretation and recommendations")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_predictions(predictor):
    """Demonstrate the prediction system - FIXED VERSION"""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: PATIENT RISK ASSESSMENT")
    print("="*80)
    
    # Test patients with different risk profiles
    test_patients = [
        {
            'name': 'Low Risk Patient (Young Female)',
            'data': {'age': 32, 'sex': 0, 'cp': 0, 'trestbps': 115, 'chol': 185, 
                    'fbs': 0, 'restecg': 0, 'thalach': 175, 'exang': 0, 
                    'oldpeak': 0.2, 'slope': 2, 'ca': 0, 'thal': 2}
        },
        {
            'name': 'Medium Risk Patient (Middle-aged Male)',
            'data': {'age': 52, 'sex': 1, 'cp': 2, 'trestbps': 145, 'chol': 235, 
                    'fbs': 0, 'restecg': 1, 'thalach': 135, 'exang': 0, 
                    'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 2}
        },
        {
            'name': 'High Risk Patient (Elderly Male)',
            'data': {'age': 68, 'sex': 1, 'cp': 3, 'trestbps': 165, 'chol': 295, 
                    'fbs': 1, 'restecg': 2, 'thalach': 115, 'exang': 1, 
                    'oldpeak': 2.8, 'slope': 0, 'ca': 2, 'thal': 7}
        }
    ]
    
    # Predict for each patient
    for patient in test_patients:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {patient['name']}")
        print(f"{'='*80}")
        
        prediction, confidence = predictor.predict_with_explanation(patient['data'])
        
        if prediction is not None:
            print(f"\n✅ Analysis completed for {patient['name']}")
        else:
            print(f"\n❌ Analysis failed for {patient['name']}")

# Run the complete system
if __name__ == "__main__":
    # Train the model
    predictor = main()
    
    if predictor:
        # Demonstrate predictions
        demo_predictions(predictor)
        
        # Interactive prediction (optional)
        print(f"\n{'='*80}")
        print("INTERACTIVE PREDICTION READY")
        print(f"{'='*80}")
        print("The system is now ready for interactive predictions!")
        print("You can use predictor.predict_with_explanation(patient_data) for detailed analysis")
        print("Or predictor.simple_predict(patient_data) for quick predictions")
        
        # Example of how to use the system
        print(f"\nExample usage:")
        print("patient = {'age': 45, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 220,")
        print("          'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,")
        print("          'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2}")
        print("result = predictor.predict_with_explanation(patient)")
    else:
        print("❌ System initialization failed")