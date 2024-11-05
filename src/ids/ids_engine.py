# File: src/ids/ids_engine.py

import os
import logging
import numpy as np
import time
import psutil
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
from src.data.cicddos_loader import CICDDoSLoader
from src.ids.ids_model import create_cnn_model, create_random_forest_model, create_xgboost_model, compile_and_fit

class IDSEngine:
    def __init__(self, config):
        self.logger = logging.getLogger("IDSEngine")
        self.config = config
        self.models = {}
        self.detection_threshold_high = config['detection_threshold_high']
        self.detection_threshold_medium = config['detection_threshold_medium']
        self.detection_threshold_low = config['detection_threshold_low']
        self.data_loader = CICDDoSLoader(config['data_path'], config['scaler_path'])
        self.data_loader.load_data()
        self.scaler = self.data_loader.scaler
        self.predictions = []
        self.model_to_use = config['model_to_use']
        self.noise_level = 0  # Initialize with no noise
        self.random_seed = config['random_seed']
        self.n_splits = 5  # for cross-validation
        self.model_paths = self._get_model_paths()
        self._load_or_train_models()
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.inference_times = []
        self.process = psutil.Process()
        self.total_predictions = 0
        self.total_time = 0

    def _get_model_paths(self):
        base_paths = {
            'cnn': self.config['cnn_model_path'],
            'rf': self.config['rf_model_path'],
            'xgb': self.config['xgb_model_path'],
        }
        
        if self.noise_level > 0:
            return {model_type: f"{os.path.splitext(path)[0]}_noise_{self.noise_level}{os.path.splitext(path)[1]}"
                    for model_type, path in base_paths.items()}
        return base_paths

    def _load_or_train_models(self):
        X_train, y_train = self.data_loader.get_train_data()
        
        for model_type in ['cnn', 'rf', 'xgb']:
            model_path = self.model_paths[model_type]
            if os.path.exists(model_path) and not self.config.get('retrain', False):
                print(f"Loading existing {model_type.upper()} model from {model_path}")
                self._load_model(model_type)
            else:
                print(f"Training new {model_type.upper()} model...")
                self.models[model_type] = self._train_model(model_type, X_train, y_train)
                self._save_model(model_type)

    def _train_model(self, model_type, X_train, y_train):
        X_train_noisy = self.add_noise(X_train)
        smote = SMOTE(random_state=self.random_seed)
        X_resampled, y_resampled = smote.fit_resample(X_train_noisy, y_train)
        
        # Ensure X_resampled is a DataFrame with named columns
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        class_weight_dict = dict(enumerate(class_weights))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        best_val_score = float('-inf')
        best_model = None

        for fold, (train_index, val_index) in enumerate(skf.split(X_resampled, y_resampled), 1):
            print(f"Training on fold {fold}/{self.n_splits}")

            # Use proper DataFrame indexing
            X_train_fold, X_val_fold = X_resampled.iloc[train_index], X_resampled.iloc[val_index]
            y_train_fold, y_val_fold = y_resampled[train_index], y_resampled[val_index]

            if model_type == 'cnn':
                # For CNN, reshape the data
                X_train_reshaped = X_train_fold.values.reshape((X_train_fold.shape[0], X_train_fold.shape[1], 1))
                X_val_reshaped = X_val_fold.values.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1))
                y_train_cat = tf.keras.utils.to_categorical(y_train_fold, num_classes=2)
                y_val_cat = tf.keras.utils.to_categorical(y_val_fold, num_classes=2)

                model = create_cnn_model((X_train_reshaped.shape[1], 1))
                history = compile_and_fit(model, X_train_reshaped, y_train_cat, X_val_reshaped, y_val_cat, class_weight_dict)
                val_score = max(history.history['val_accuracy'])
            else:
                if model_type == 'rf':
                    model = create_random_forest_model()
                elif model_type == 'xgb':
                    model = create_xgboost_model()

                model.fit(
                    X_train_fold, y_train_fold,
                    sample_weight=[class_weight_dict[y] for y in y_train_fold]
                )
                val_score = model.score(X_val_fold, y_val_fold)

            print(f"Validation score for fold {fold}: {val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model

        return best_model

    def _save_model(self, model_type):
        model_path = self.model_paths[model_type]
        if model_type == 'cnn':
            self.models[model_type].save(model_path)
        else:
            joblib.dump(self.models[model_type], model_path)
        self.logger.info(f"Saved {model_type.upper()} model to {model_path}")

    def get_test_data(self):
        return self.data_loader.get_test_data()

    def _load_model(self, model_type):
        model_path = self.model_paths[model_type]
        if model_type == 'cnn':
            self.models[model_type] = tf.keras.models.load_model(model_path)
        else:
            self.models[model_type] = joblib.load(model_path)

    def set_noise_level(self, noise_level):
        if noise_level != self.noise_level:
            self.noise_level = noise_level
            self.model_paths = self._get_model_paths()
            self._load_or_train_models()
            self.logger.info(f"Noise level set to {noise_level}")

    def add_noise(self, X):
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, X.shape)
            return X + noise
        return X

    def detect(self, features, true_label=None):
        """
        Modified detect method with performance tracking
        """
        # Start timing and memory tracking
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Prepare input
        X = np.array(features).reshape(1, -1)
        X_noisy = self.add_noise(X)

        # Model prediction
        try:
            if self.model_to_use == 'cnn':
                X_reshaped = X_noisy.reshape((X_noisy.shape[0], X_noisy.shape[1], 1))
                pred_proba = self.models['cnn'].predict(X_reshaped)[0][1]
            else:
                pred_proba = self.models[self.model_to_use].predict_proba(X_noisy)[0][1]

            # End timing and calculate metrics
            end_time = time.perf_counter()
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            inference_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            
            # Store performance metrics
            self.inference_times.append(inference_time)
            self.total_predictions += 1
            self.total_time += inference_time

            # Calculate prediction
            pred_label = int(pred_proba >= 0.5)
            risk_level = self._get_risk_level(pred_proba)

            # Store predictions if true label is provided
            if true_label is not None:
                self.y_true.append(true_label)
                self.y_pred.append(pred_label)
                self.y_pred_proba.append(pred_proba)

            performance_metrics = {
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'total_predictions': self.total_predictions,
                'average_time': np.mean(self.inference_times)
            }

            self.logger.debug(
                f"Detection performed - Time: {inference_time*1000:.2f}ms, "
                f"Memory: {memory_usage:.2f}MB, "
                f"Probability: {pred_proba:.3f}, "
                f"Risk Level: {risk_level}"
            )

            return risk_level, pred_proba, performance_metrics

        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
            return 'error', 0.0, None


    def _get_risk_level(self, probability):
        if probability >= self.detection_threshold_high:
            return 'high'
        elif probability >= self.detection_threshold_medium:
            return 'medium'
        elif probability >= self.detection_threshold_low:
            return 'low'
        else:
            return 'very_low'

    def get_predictions(self):
        return self.predictions

    def clear_predictions(self):
        self.predictions = []

    def evaluate(self, X_test=None, y_test=None):
        """
        Enhanced evaluate method with performance metrics
        """
        if X_test is None or y_test is None:
            X_test, y_test = self.get_test_data()
        
        # Performance tracking for evaluation
        eval_start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Add noise to test data
        X_test_noisy = self.add_noise(X_test)
        
        # Model evaluation
        if self.model_to_use == 'cnn':
            X_reshaped = X_test_noisy.reshape((X_test_noisy.shape[0], X_test_noisy.shape[1], 1))
            y_pred_proba = self.models['cnn'].predict(X_reshaped)
            y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
        else:
            y_pred = self.models[self.model_to_use].predict(X_test_noisy)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate performance metrics
        eval_end_time = time.perf_counter()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        evaluation_time = eval_end_time - eval_start_time
        memory_usage = final_memory - initial_memory
        predictions_per_second = len(X_test) / evaluation_time
        
        results = {
            # Detection Performance
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            
            # Resource Usage
            'evaluation_time_seconds': evaluation_time,
            'memory_usage_mb': memory_usage,
            'predictions_per_second': predictions_per_second,
            'average_inference_time_ms': (evaluation_time / len(X_test)) * 1000
        }
        
        self.logger.info(
            f"Model Evaluation Completed - "
            f"Accuracy: {accuracy:.3f}, "
            f"Time: {evaluation_time:.2f}s, "
            f"Throughput: {predictions_per_second:.1f} pred/s"
        )
        
        return results
    
    def get_evaluation_metrics(self):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        if not self.y_true or not self.y_pred or not self.y_pred_proba:
            self.logger.warning("No predictions have been made yet.")
            return {}

        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
            metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
            metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
            metrics['auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
        except Exception as e:
            self.logger.error(f"Error calculating evaluation metrics: {str(e)}")
        return metrics

    
    def get_performance_metrics(self):
        """
        Get comprehensive performance metrics for the IDS
        """
        if not self.inference_times:
            return {
                'average_inference_time_ms': 0,
                'max_inference_time_ms': 0,
                'total_predictions': 0,
                'predictions_per_second': 0,
                'memory_usage_mb': 0
            }

        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        return {
            'average_inference_time_ms': np.mean(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'total_predictions': self.total_predictions,
            'predictions_per_second': self.total_predictions / max(self.total_time, 1),
            'memory_usage_mb': current_memory
        }