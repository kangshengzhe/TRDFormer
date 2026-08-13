import argparse
import csv
import time
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ==================== 自动检测项目根目录 ====================
# 根据当前脚本位置向上查找项目根目录
# 脚本位置: <project>/scripts/train/pre_wind_iTransformer_LSTM.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # scripts/train/
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)                       # scripts/
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)                     # 项目根目录

# 设置项目根目录为工作目录
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"[INFO] Project root: {PROJECT_ROOT}")
print(f"[INFO] Working directory: {os.getcwd()}")

# 环境检测
KAGGLE_MODE = os.path.exists('/kaggle/input')
if KAGGLE_MODE:
    print("[INFO] Running in Kaggle environment")
else:
    print("[INFO] Running in local environment")

if __name__ == "__main__":
    # 导入模块（必须在设置sys.path之后）
    import dill
    import numpy as np
    import torch
    from matplotlib import pyplot as plt
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from models import iTransformer_LSTM
    from data import load_wind_data, data_detime
    from utils.tools import EarlyStopping, same_seeds, train, evaluate
    
    # 固定随机种子
    seeds = 42
    same_seeds(seeds)
    
    # 风电数据配置（使用相对路径）
    file_path = os.path.join(PROJECT_ROOT, 'data/wind/sdwpf_turb1_cleaned_final.csv')
    dataset = 'Wind_Turb1'
    
    # 检查数据文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # 超参数配置
    parser = argparse.ArgumentParser(description="Wind Power Forecasting")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    
    # 时间窗口配置（10分钟间隔）
    time_length = 144  # 24小时 = 144个10分钟间隔
    predict_length = 1  # 预测下一步
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载风电数据
    print(f"Loading wind data from: {file_path}")
    data_train, data_valid, data_test, timestamp_train, timestamp_valid, timestamp_test, scalar = load_wind_data(
        file_path, train_ratio=0.8, test_ratio=0.1, lookback_length=time_length
    )
    
    multi_steps = False
    
    # 创建数据集
    dataset_train = data_detime(data=data_train, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    dataset_valid = data_detime(data=data_valid, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    dataset_test = data_detime(data=data_test, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    
    # 创建数据加载器
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # 模型配置
    params_dict = {'hidden_dim': 32, 'layer_L': 3, 'layer_I': 4, 'heads': 12, 'dim_lstm': 32}
    model = iTransformer_LSTM(
        input_size=5,  # Patv + Wspd + Wdir + Etmp + Itmp
        length_input=time_length,
        dim_embed=params_dict['hidden_dim'],
        dim_lstm=params_dict['dim_lstm'],
        depth=params_dict['layer_I'],
        heads=params_dict['heads'],
        depth_lstm=params_dict['layer_L']
    ).to(device)
    
    print(f"Model parameters: {params_dict}")
    print(f"Input features: 5 (Patv, Wspd, Wdir, Etmp, Itmp)")
    print(f"Time window: {time_length} steps (24 hours)")
    
    # 损失函数和优化器
    criterion_MAE = nn.L1Loss(reduction='sum').to(device)
    criterion_MSE = nn.MSELoss(reduction='sum').to(device)
    optm = optim.Adam(model.parameters(), lr=learning_rate)
    optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optm, mode="min", factor=0.5, patience=5
    )
    
    # 模型保存路径（使用相对路径）
    model_name = f"iTransformer_LSTM_wind_{dataset}"
    model_save_dir = os.path.join(PROJECT_ROOT, f"model_save/wind")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save = os.path.join(model_save_dir, f"{model_name}.pt")
    
    # 训练记录
    train_losses, valid_losses = [], []
    earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)
    
    # 是否需要训练
    need_train = True  # 设置为True开始训练
    
    if need_train:
        print(f"\nStarting training: {model_name}")
        print("=" * 80)
        
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
                print(f'Epoch {epoch+1}/{epochs} | Time: {(time.time() - time_start):.2f}s')
                print(f'  Train Loss: {train_loss:.4f} | LR: {optm.state_dict()["param_groups"][0]["lr"]:.6f}')
                print(f'  Valid Loss: {valid_loss:.4f}')
                print(f'  MAE: {ms[0]:.4f} | RMSE: {ms[1]:.4f} | R²: {ms[2]:.4f} | MBE: {ms[3]:.4f}')
                print("-" * 80)
                
                if earlystopping.early_stop:
                    print("Early stopping triggered")
                    break
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
        plt.plot(np.arange(len(valid_losses)), valid_losses, label="Valid Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Curve - {model_name}")
        plt.grid(True)
        plt.savefig(f"{model_save_dir}/training_curve.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    # 加载最佳模型进行测试
    print(f"\nLoading best model from: {model_save}")
    with open(model_save, "rb") as f:
        model = torch.load(f, pickle_module=dill)
    model = model.to(device)
    
    # 测试集评估
    test_loss, ms_test = evaluate(
        data=test_loader, model=model, criterion=criterion_MAE, device=device, scalar=scalar
    )
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'MAE:  {ms_test[0]:.4f}')
    print(f'RMSE: {ms_test[1]:.4f}')
    print(f'R²:   {ms_test[2]:.4f}')
    print(f'MBE:  {ms_test[3]:.4f}')
    print("=" * 80)
    
    # 保存测试结果（使用相对路径）
    records_dir = os.path.join(PROJECT_ROOT, f"data_record/wind")
    os.makedirs(records_dir, exist_ok=True)
    
    with open(f'{records_dir}/Metrics_{model_name}.csv', 'a', encoding='utf-8', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([model_name, ms_test[0], ms_test[1], ms_test[2], ms_test[3]])
    
    print(f"\nResults saved to: {records_dir}/Metrics_{model_name}.csv")
