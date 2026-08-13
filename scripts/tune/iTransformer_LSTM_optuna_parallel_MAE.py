import argparse
import csv
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import optuna
from data import load_wind_data, data_detime
from models import iTransformer_LSTM
from utils.tools import metrics_of_pv, EarlyStopping, same_seeds, train, evaluate
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    seeds = 42
    same_seeds(seeds)

    parser = argparse.ArgumentParser(description="Wind Power Hyperparameter Tuning")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=150)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    # 风电数据配置
    file_path = './data/wind/sdwpf_turb1_cleaned_final.csv'
    dataset = 'Wind_Turb1'
    time_length = 144  # 24小时 = 144个10分钟间隔
    predict_length = 1
    device = torch.device('cuda:0')
    multi_steps = False
    epoch = 0

    # 加载风电数据
    data_train, data_valid, data_test, timestamp_train, timestamp_valid, timestamp_test, scalar = load_wind_data(
        file_path, train_ratio=0.8, test_ratio=0.1, lookback_length=time_length
    )

    dataset_train = data_detime(data=data_train, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    dataset_valid = data_detime(data=data_valid, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    dataset_test = data_detime(data=data_test, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    def objective(trial):
        dim_embed = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])
        layer_L = trial.suggest_categorical('layer_L', [1, 2, 3, 4])
        layer_I = trial.suggest_categorical('layer_I', [1, 2, 3, 4, 5, 6])
        heads = trial.suggest_categorical('heads', [2, 4, 6, 8, 12, 16])
        dim_lstm = trial.suggest_categorical('dim_lstm', [16, 32, 64, 128, 256])

        model = iTransformer_LSTM(
            input_size=5,  # Patv, Wspd, Wdir, Etmp, Itmp
            length_input=time_length,
            dim_lstm=dim_lstm,
            depth_lstm=layer_L,
            dim_embed=dim_embed,
            depth=layer_I,
            heads=heads
        ).to(device)

        criterion_MAE = nn.L1Loss(reduction='sum').to(device)
        optm = optim.Adam(model.parameters(), lr=learning_rate)
        optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optm, mode="min", factor=0.5, patience=5
        )

        model_name = f"iTransformer_LSTM_optuna_{dataset}"
        model_save = f"model_save/wind/{model_name}.pt"
        train_losses, valid_losses = [], []
        earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)

        need_train = True

        if need_train:
            try:
                for epoch in range(epochs):
                    time_start = time.time()
                    train_loss = train(data=train_loader, model=model, criterion=criterion_MAE, optm=optm, device=device)
                    valid_loss, ms = evaluate(data=valid_loader, model=model, criterion=criterion_MAE, device=device)

                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    optm_schedule.step(valid_loss)
                    earlystopping(valid_loss, model)

                    print('')
                    print(f'{model_name}|time:{(time.time() - time_start):.2f}|Loss_train:{train_loss:.4f}|Learning_rate:{optm.state_dict()["param_groups"][0]["lr"]:.4f}\n'
                          f'Loss_valid:{valid_loss:.4f}|MAE:{ms[0]:.4f}|RMSE:{ms[1]:.4f}|R2:{ms[2]:.4f}|MBE:{ms[3]:.4f}', flush=True)

                    if earlystopping.early_stop:
                        print("Early stopping")
                        break
            except KeyboardInterrupt:
                print("Training interrupted by user")

        test_loss, ms_test = evaluate(
            data=test_loader, model=model, criterion=criterion_MAE, device=device
        )

        print(f'Test_valid:{test_loss:.4f}|MAE:{ms_test[0]:.4f}|RMSE:{ms_test[1]:.4f}|R2:{ms_test[2]:.4f}|MBE:{ms_test[3]:.4f}')

        with open(f'data_record/wind/Metrics_{model_name}.csv', 'a', encoding='utf-8', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([f'{model_name}', ms_test[0], ms_test[1], ms_test[2], ms_test[3]])

        return test_loss


    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
        storage='sqlite:///db.sqlite3',
        study_name=f'wind_{dataset}_optuna'
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params, '\n', study.best_value)
