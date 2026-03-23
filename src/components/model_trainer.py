import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        try:
            report = {}
            trained_models = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                para = param[model_name]

                # GridSearchCV
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    n_jobs=-1,
                    verbose=1
                )

                # Train
                gs.fit(X_train, y_train)

                # Best model
                best_model = gs.best_estimator_

                # Save trained model
                trained_models[model_name] = best_model

                # Predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Scores
                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_score

                logging.info(f"{model_name} → Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

            return report, trained_models

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.8],
                    'n_estimators': [16, 32, 64, 128]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05],
                    'n_estimators': [16, 32, 64, 128]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [.1, .01],
                    'iterations': [30, 50]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01],
                    'n_estimators': [16, 32, 64, 128]
                }
            }

            # ✅ Evaluate models
            model_report, trained_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # ✅ Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy")

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score:.4f}")

            # ✅ Save best trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ✅ Final prediction
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)