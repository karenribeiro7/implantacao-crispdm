import os
import argparse
import shutil

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import mlflow
import mlflow.sklearn

def load_data(train_path: str):
    df = pd.read_csv(train_path)

    # pré-processamento 
    df = df.drop(columns=['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'Approved_Conversion'])
    df["age"] = df["age"].replace({"30-34":1 ,"45-49":2 ,"35-39":3, "40-44":4})
    df["gender"] = df["gender"].replace({"M":1 ,"F":2})

    X = df.drop(columns=['Total_Conversion'])
    y = df['Total_Conversion']
    return X, y

def train_and_log_models(train_path: str, experiment_name: str = "conversion_rate_experiments"):
    mlflow.set_experiment(experiment_name)

    X, y = load_data(train_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    categorical_cols = ['interest']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    resultados = []          
    modelos_para_servir = [] 

    # Ridge
    with mlflow.start_run(run_name="RidgeRegression"):
        scaler = StandardScaler()
        encoder = OneHotEncoder(drop='first', sparse_output=False)

        X_train_num = scaler.fit_transform(X_train[numerical_cols])
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_train_proc = np.hstack([X_train_num, X_train_cat])

        X_test_num = scaler.transform(X_test[numerical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        X_test_proc = np.hstack([X_test_num, X_test_cat])

        model = Ridge()
        param_grid = {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X_train_proc, y_train)

        best_model = grid.best_estimator_

        # métricas no teste
        y_pred = best_model.predict(X_test_proc)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "Ridge")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # pipeline com pré-processamento para servir
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
            ]
        )
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(**grid.best_params_))
        ])
        pipe.fit(X_train, y_train)

        mlflow.sklearn.log_model(pipe, "model")

        resultados.append(("Ridge", rmse, mae, r2))
        modelos_para_servir.append(("Ridge", rmse, pipe))

    
    # Decision Tree
    with mlflow.start_run(run_name="DecisionTreeRegressor"):
        model = DecisionTreeRegressor(random_state=42)
        param_grid = {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        mlflow.sklearn.log_model(best_model, "model")

        resultados.append(("DecisionTree", rmse, mae, r2))
        modelos_para_servir.append(("DecisionTree", rmse, best_model))

    # Random Forest
    with mlflow.start_run(run_name="RandomForestRegressor"):
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        mlflow.sklearn.log_model(best_model, "model")

        resultados.append(("RandomForest", rmse, mae, r2))
        modelos_para_servir.append(("RandomForest", rmse, best_model))

    
    # MLPRegressor
    with mlflow.start_run(run_name="MLPRegressor"):
        scaler = StandardScaler()
        encoder = OneHotEncoder(drop='first', sparse_output=False)

        X_train_num = scaler.fit_transform(X_train[numerical_cols])
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
        X_train_proc = np.hstack([X_train_num, X_train_cat])

        X_test_num = scaler.transform(X_test[numerical_cols])
        X_test_cat = encoder.transform(X_test[categorical_cols])
        X_test_proc = np.hstack([X_test_num, X_test_cat])

        model = MLPRegressor(max_iter=1000, random_state=42)

        param_grid = {
            "hidden_layer_sizes": [(32,), (64,), (64, 32)],
            "learning_rate_init": [0.001, 0.01],
            "alpha": [0.0001, 0.001]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X_train_proc, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test_proc)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "MLPRegressor")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # pipeline com pré-processamento para servir
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
            ]
        )
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", MLPRegressor(
                max_iter=1000,
                random_state=42,
                **{k: v for k, v in grid.best_params_.items()}
            ))
        ])
        pipe.fit(X_train, y_train)

        mlflow.sklearn.log_model(pipe, "model")

        resultados.append(("MLPRegressor", rmse, mae, r2))
        modelos_para_servir.append(("MLPRegressor", rmse, pipe))

    # GradientBoostingRegressor
    with mlflow.start_run(run_name="GradientBoostingRegressor"):
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [1.0, 0.8]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        mlflow.sklearn.log_model(best_model, "model")

        resultados.append(("GradientBoosting", rmse, mae, r2))
        modelos_para_servir.append(("GradientBoosting", rmse, best_model))

    # selecionar melhor modelo pelo menor RMSE e salvar
    resultados_df = pd.DataFrame(resultados, columns=["modelo", "rmse", "mae", "r2"])
    print("\nResultados (teste):")
    print(resultados_df)

    best_row = resultados_df.loc[resultados_df["rmse"].idxmin()]
    best_model_name = best_row["modelo"]
    print("\nMelhor modelo:", best_model_name)

    # encontrar o objeto do melhor modelo
    best_model_obj = None
    best_rmse = best_row["rmse"]
    for nome, rmse_val, modelo_obj in modelos_para_servir:
        if nome == best_model_name and abs(rmse_val - best_rmse) < 1e-8:
            best_model_obj = modelo_obj
            break

    os.makedirs("/app/models", exist_ok=True)
    save_path = "/app/models/best_model"

    if os.path.exists(save_path):
            shutil.rmtree(save_path)

    print(f"Salvando melhor modelo em: {save_path}")
    mlflow.sklearn.save_model(best_model_obj, path=save_path)

    train_flag_path = "/app/models/train_done.flag"
    with open(train_flag_path, "w") as f:
        f.write("done")

    return resultados_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=str,
        default="../data/KAG_conversion_data.csv",
    )
    args = parser.parse_args()

    train_and_log_models(args.train_path)