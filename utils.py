import os
import torch
import torchmetrics.functional.classification as tmf
from typing import Tuple
import pandas as pd
import wandb
import requests
import zipfile
import time
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    wrote = 0

    with open(dest_path, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
    ) as progress_bar:
        start_time = time.time()
        for data in response.iter_content(block_size):
            wrote += len(data)
            file.write(data)
            progress_bar.update(len(data))
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                speed = wrote / elapsed_time / 1024  # Speed in KB/s
                progress_bar.set_postfix(speed=f'{speed:.2f} KB/s')

    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")

def download_esc50(root_dir:str=None)->str:
    '''
    下载ESC-50数据集，并且解压到root_dir目录下
    :param root_dir: 解压目录，默认为None，即解压到当前目录
    :return: esc50数据集路径
    '''
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    if root_dir is None:
        root_dir = os.getcwd()
    # 使用requests下载，加入进度条显示下载进度
    print("Downloading ESC-50 dataset...")
    zip_file = os.path.join(root_dir, "ESC-50.zip")
    download_file(url=url, dest_path=zip_file)    
    # 读取zip文件
    with zipfile.ZipFile(zip_file) as z:
        # 解压到指定目录
        z.extractall(root_dir)
    # 删除zip文件
    os.remove(zip_file)
    # 返回解压后的目录
    return os.path.join(root_dir, "ESC-50-master")

def log2wandb(**kwargs)->None: # 包装wandb.log函数，使用关键字参数传入参数
    '''
    使用关键字参数将参数传入wandb.log函数
    '''
    wandb.log(kwargs)

def df2table(df:pd.DataFrame)->wandb.Table:
    '''
    将pandas数据框转换为wandb.Table对象
    :param df: pandas数据框
    :return: wandb.Table对象
    '''
    table = wandb.Table(columns=list(df.columns))
    for i, row in df.iterrows():
        table.add_data(*row.tolist())
    return table

def train_an_epoch(model, optimizer, scheduler, data_loader, device, loss_func, tqdm_instance=None) -> float:
    '''
    训练一个epoch
    :param model: 模型
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param data_loader: 数据加载器
    :param device: 设备
    :param loss_func: 损失函数
    :param tqdm_instance: tqdm实例，用于显示进度条
    :return: 训练损失
    '''
    training_loss, _ = 0.0, model.train()  # 初始化

    for count, (features, labels, _) in enumerate(data_loader, start=1):

        features, labels = features.to(device), labels.to(device)  # 加载到设备

        optimizer.zero_grad()  # 清空梯度

        outputs = model(features)  # 前向传播
        outputs = torch.softmax(outputs, dim=1)  # 转换为概率形式

        loss = loss_func(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播

        optimizer.step()  # 更新参数
        scheduler.step()  # 更新学习率

        training_loss += loss.item()
        if tqdm_instance is not None:
            tqdm_instance.set_description(
                f"[train] loss: {training_loss/count:.4f} Progress: {count}/{len(data_loader)}"
            )

    return training_loss / len(data_loader)

def test_an_epoch(categoties, model, data_loader, device, loss_func, tqdm_instance=None) -> Tuple[float, float, pd.DataFrame]:
    '''
    测试一个epoch
    :param categoties: 类别列表
    :param model: 模型
    :param data_loader: 数据加载器
    :param device: 设备
    :param loss_func: 损失函数
    :param tqdm_instance: tqdm实例，用于显示进度条
    :return: 损失，准确率，错误样本
    '''
    total_loss, bad_cases, outputs_lst, targets_lst, _ = 0, [], [], [], model.eval() # 初始化

    with torch.no_grad():
        for count, (features, labels, idxes) in enumerate(data_loader, start=1):
            features, labels = features.to(device), labels.to(device)  # 加载到设备
            outputs = model(features)  # 预测
            outputs = torch.softmax(outputs, dim=1)  # 转换为概率形式

            loss = loss_func(outputs, labels)  # 计算损失
            total_loss += loss.item()  # 累计损失
            outputs_lst.append(outputs), targets_lst.append(labels)  # 记录输出和目标
            pred = outputs.argmax(axis=1)  # 预测类别

            for i, result in enumerate(pred == labels):
                if not result:
                    bad_cases.append({
                        "idx": idxes[i].item(),
                        "label": labels[i].item(),
                        "pred_label": pred[i].item(),
                        "label_name": categoties[labels[i].item()],
                        "pred_name": categoties[pred[i].item()],
                    })  # 记录错误样本
                    
            if tqdm_instance is not None:
                tqdm_instance.set_description(f"[valid] Progress: {count}/{len(data_loader)}")
                
    accuracy = tmf.accuracy(
        preds=torch.argmax(torch.cat(outputs_lst, dim=0), dim=1),
        target=torch.cat(targets_lst, dim=0),
        task="multiclass",
        num_classes=len(categoties)
    )  # 计算准确率
    return total_loss / len(data_loader), accuracy.item(), pd.DataFrame(bad_cases)

def main():
    download_esc50('test')
    print('ok')
    
if __name__ == '__main__':
    main()