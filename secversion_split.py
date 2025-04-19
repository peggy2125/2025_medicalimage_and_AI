import os
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
import logging
from torchvision import transforms
import random
from pathlib import Path
import seaborn as sns

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MURA-Splitter")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MURA数据集平衡分割工具')
    parser.add_argument('--data_root', type=str, default='MURA-v1.1', 
                        help='MURA数据集根目录')
    parser.add_argument('--train_csv', type=str, default='MURA-v1.1/train_labeled_studies.csv',
                        help='原始训练集CSV路径')
    parser.add_argument('--valid_csv', type=str, default='MURA-v1.1/valid_labeled_studies.csv',
                        help='原始验证集CSV路径')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='输出目录')
    parser.add_argument('--valid_ratio', type=float, default=0.2,
                        help='验证集占比')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='每个组内的聚类数量')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--min_valid_samples', type=int, default=2,
                        help='每个聚类的最小验证样本数')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像预处理尺寸')
    parser.add_argument('--save_full_info', action='store_true',
                        help='是否保存完整特征信息（用于分析）')
    parser.add_argument('--patient_level', action='store_true', default=True,
                        help='以病人为单位进行分割，避免数据泄漏')
    return parser.parse_args()

def setup_environment(args):
    """设置环境和随机种子"""
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def get_transform(img_size):
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1)
    ])

def extract_features(img_path, transform):
    """提取单个图像的特征"""
    try:
        img = Image.open(img_path).convert("L")
        img = transform(img)
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # 基本统计特征
        pixel_mean = img_np.mean()
        pixel_std = img_np.std()
        
        # 直方图特征 (简化版)
        hist, _ = np.histogram(img_np, bins=10, range=(0, 1))
        hist = hist / hist.sum()  # 归一化
        
        # 纹理特征 (简化版 - 标准差的标准差作为粗略的纹理指标)
        patches = [img_np[i:i+32, j:j+32] for i in range(0, img_np.shape[0], 32) 
                                           for j in range(0, img_np.shape[1], 32)]
        patch_stds = [p.std() for p in patches if p.size > 0]
        texture = np.std(patch_stds) if patch_stds else 0
        
        return {
            'mean': pixel_mean,
            'std': pixel_std,
            'texture': texture,
            'hist': hist.tolist()
        }
    except Exception as e:
        logger.warning(f"处理图像 {img_path} 时出错: {e}")
        # 返回默认值
        return {
            'mean': 0.5,
            'std': 0.1,
            'texture': 0.0,
            'hist': [0.1] * 10
        }

def get_patient_stats(csv_path, data_root, transform, mode):
    """获取患者统计信息"""
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到CSV文件: {csv_path}")
        
        df = pd.read_csv(csv_path, header=None, names=['path', 'label'])
        stats = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{mode}数据"):
            study_rel_path, label = row['path'], row['label']
            study_path = os.path.join(data_root, study_rel_path)
            
            if not os.path.exists(study_path):
                logger.warning(f"找不到路径: {study_path}, 跳过")
                continue
                
            image_paths = glob(os.path.join(study_path, "*.png"))
            if not image_paths:
                logger.warning(f"目录不包含PNG图像: {study_path}, 跳过")
                continue

            # 提取身体部位
            body_part = study_rel_path.split("/")[2]  # 假设格式是 "MURA-v1.1/train/XR_PART/..."
            
            # 收集所有图像的特征
            features_list = []
            for img_path in image_paths:
                features = extract_features(img_path, transform)
                features_list.append(features)
            
            # 计算研究级别的特征
            if features_list:
                means = [f['mean'] for f in features_list]
                stds = [f['std'] for f in features_list]
                textures = [f['texture'] for f in features_list]
                hist_avg = np.mean([f['hist'] for f in features_list], axis=0).tolist()
                
                patient_mean = np.mean(means)
                patient_std = np.mean(stds)
                patient_texture = np.mean(textures)
                
                # 更多统计信息
                img_count = len(image_paths)
                
                stats.append({
                    "path": study_rel_path,
                    "body_part": body_part,
                    "label": label,
                    "mean": patient_mean,
                    "std": patient_std,
                    "texture": patient_texture,
                    "hist": hist_avg,
                    "img_count": img_count,
                    "mode": mode
                })
        
        return pd.DataFrame(stats)
    except Exception as e:
        logger.error(f"处理CSV {csv_path} 时出错: {e}")
        return pd.DataFrame()

def extract_patient_id(path):
    """从路径中提取病人ID"""
    # 假设路径格式为 "MURA-v1.1/train/XR_PART/patient_XXXX/study_Y/image.png"
    # 我们提取 patient_XXXX 作为病人ID
    parts = path.split('/')
    for part in parts:
        if part.startswith('patient'):
            return part
    return None

def stratified_cluster_split(df, args):
    """使用分层聚类进行训练集和验证集分割，确保同一病人不会被分到不同集合"""
    logger.info("开始分层聚类分割...")
    
    # 特征归一化
    scaler = StandardScaler()
    feature_cols = ['mean', 'std', 'texture']
    
    # 添加病人ID
    df['patient_id'] = df['path'].apply(extract_patient_id)
    
    df_train_new = []
    df_valid_new = []
    
    # 按身体部位分层
    for body_part, body_group in df.groupby('body_part'):
        logger.info(f"处理身体部位: {body_part}")
        
        # 再按标签分层
        for label, label_group in body_group.groupby('label'):
            # 按病人ID进行聚合，确保同一病人的所有数据保持在一起
            patient_groups = []
            
            if args.patient_level:
                # 获取该组中所有唯一的病人ID
                unique_patients = label_group['patient_id'].unique()
                logger.info(f"部位 {body_part}, 标签 {label} 的唯一病人数: {len(unique_patients)}")
                
                # 按病人ID聚合数据
                for patient_id in unique_patients:
                    patient_data = label_group[label_group['patient_id'] == patient_id]
                    if not patient_data.empty:
                        # 取该病人的平均特征作为聚类依据
                        patient_features = {
                            'patient_id': patient_id,
                            'mean': patient_data['mean'].mean(),
                            'std': patient_data['std'].mean(),
                            'texture': patient_data['texture'].mean(),
                            'hist': np.mean(np.array(patient_data['hist'].tolist()), axis=0).tolist(),
                            'data': patient_data
                        }
                        patient_groups.append(patient_features)
            else:
                # 不按病人聚合，直接使用样本
                for _, row in label_group.iterrows():
                    patient_groups.append({
                        'patient_id': row['patient_id'],
                        'mean': row['mean'],
                        'std': row['std'],
                        'texture': row['texture'],
                        'hist': row['hist'],
                        'data': label_group[label_group['patient_id'] == row['patient_id']]
                    })
            
            # 确定聚类数
            if len(patient_groups) < args.n_clusters:
                logger.warning(f"部位 {body_part}, 标签 {label} 的病人数 ({len(patient_groups)}) 小于聚类数 ({args.n_clusters})，减少聚类数")
                n_clusters = max(2, len(patient_groups) // 2)  # 至少2个聚类，或者病人数的一半
            else:
                n_clusters = args.n_clusters
                
            # 准备特征矩阵
            X = np.array([[pg['mean'], pg['std'], pg['texture']] for pg in patient_groups])
            if len(X) == 0:
                logger.warning(f"部位 {body_part}, 标签 {label} 没有有效数据，跳过")
                continue
                
            X_scaled = scaler.fit_transform(X)
            
            # 使用直方图特征补充
            hist_features = np.array([pg['hist'] for pg in patient_groups])
            combined_features = np.hstack([X_scaled, hist_features])
            
            # PCA降维（如果样本足够）
            if len(combined_features) > 5:  # 至少需要几个样本才能进行PCA
                pca = PCA(n_components=min(0.95, 1.0))  # 确保不会超过样本数
                X_pca = pca.fit_transform(combined_features)
            else:
                X_pca = combined_features
            
            # 聚类
            clusters = []
            if len(X_pca) > n_clusters:
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    random_state=args.random_seed,
                    n_init=10
                )
                cluster_ids = kmeans.fit_predict(X_pca)
                
                # 将聚类结果添加到病人组
                for i, pg in enumerate(patient_groups):
                    pg['cluster'] = cluster_ids[i]
                    
                # 按聚类ID分组
                for cluster_id in range(n_clusters):
                    clusters.append([pg for pg in patient_groups if pg['cluster'] == cluster_id])
            else:
                # 样本太少，每个病人一个聚类
                for i, pg in enumerate(patient_groups):
                    pg['cluster'] = i
                    clusters.append([pg])
            
            # 从每个聚类中抽取训练集和验证集
            for cluster_idx, cluster in enumerate(clusters):
                if not cluster:  # 空聚类
                    continue
                    
                # 计算验证集大小
                cluster_size = len(cluster)
                valid_size = max(
                    min(args.min_valid_samples, cluster_size // 2),
                    int(cluster_size * args.valid_ratio)
                )
                valid_size = min(valid_size, cluster_size - 1)  # 确保至少有1个样本用于训练
                
                if valid_size <= 0 or cluster_size <= 1:
                    # 样本太少，全部用于训练
                    train_patients = cluster
                    valid_patients = []
                else:
                    # 随机抽样
                    random.shuffle(cluster)
                    valid_patients = cluster[:valid_size]
                    train_patients = cluster[valid_size:]
                
                # 收集训练和验证数据
                for pg in train_patients:
                    df_train_new.append(pg['data'])
                
                for pg in valid_patients:
                    df_valid_new.append(pg['data'])
                
                # 记录分配情况
                train_samples = sum(len(pg['data']) for pg in train_patients)
                valid_samples = sum(len(pg['data']) for pg in valid_patients)
                logger.info(f"部位: {body_part}, 标签: {label}, 聚类: {cluster_idx}, "
                           f"总病人: {cluster_size}, 训练病人: {len(train_patients)}, 验证病人: {len(valid_patients)}, "
                           f"训练样本: {train_samples}, 验证样本: {valid_samples}")
    
    # 合并结果
    df_train_final = pd.concat(df_train_new, ignore_index=True) if df_train_new else pd.DataFrame()
    df_valid_final = pd.concat(df_valid_new, ignore_index=True) if df_valid_new else pd.DataFrame()
    
    # 确认没有病人同时出现在训练集和验证集
    if not df_train_final.empty and not df_valid_final.empty:
        train_patients = set(df_train_final['patient_id'])
        valid_patients = set(df_valid_final['patient_id'])
        overlap = train_patients.intersection(valid_patients)
        
        if overlap:
            logger.warning(f"发现 {len(overlap)} 个病人同时出现在训练集和验证集中！")
            for patient in overlap:
                logger.warning(f"重叠病人: {patient}")
        else:
            logger.info("✓ 验证完成：没有病人同时出现在训练集和验证集中")
    
    return df_train_final, df_valid_final

def save_results(df_train, df_valid, args):
    """保存结果到CSV文件（只保存path和label）"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 提取需要的列
    df_train_csv = df_train[['path', 'label']]
    df_valid_csv = df_valid[['path', 'label']]
    
    # 保存CSV - 只包含path和label，无表头
    train_csv_path = os.path.join(args.output_dir, 'train_balanced_split.csv')
    valid_csv_path = os.path.join(args.output_dir, 'valid_balanced_split.csv')
    
    df_train_csv.to_csv(train_csv_path, index=False, header=False)
    df_valid_csv.to_csv(valid_csv_path, index=False, header=False)
    
    logger.info(f"已保存训练集CSV: {train_csv_path}")
    logger.info(f"已保存验证集CSV: {valid_csv_path}")
    
    # 可选：保存完整信息用于分析（不用于训练）
    if args.save_full_info:
        train_full_path = os.path.join(args.output_dir, 'train_full_info.csv')
        valid_full_path = os.path.join(args.output_dir, 'valid_full_info.csv')
        
        # 转换hist列为字符串以便保存
        df_train['hist_str'] = df_train['hist'].apply(lambda x: ','.join(map(str, x)))
        df_valid['hist_str'] = df_valid['hist'].apply(lambda x: ','.join(map(str, x)))
        
        # 删除原始hist列
        df_train_full = df_train.drop('hist', axis=1)
        df_valid_full = df_valid.drop('hist', axis=1)
        
        df_train_full.to_csv(train_full_path, index=False)
        df_valid_full.to_csv(valid_full_path, index=False)
        
        logger.info(f"已保存训练集完整信息: {train_full_path}")
        logger.info(f"已保存验证集完整信息: {valid_full_path}")

def visualize_distribution(df_train, df_valid, args):
    """可视化训练集和验证集的分布"""
    # 设置绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 图1: 不同身体部位的样本数量分布
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    body_parts = sorted(df_train['body_part'].unique())
    train_count = df_train['body_part'].value_counts().reindex(body_parts).fillna(0)
    valid_count = df_valid['body_part'].value_counts().reindex(body_parts).fillna(0)
    
    # 绘制条形图
    x = np.arange(len(body_parts))
    width = 0.35
    
    plt.bar(x - width/2, train_count, width, label='训练集')
    plt.bar(x + width/2, valid_count, width, label='验证集')
    
    plt.ylabel('样本数量')
    plt.title('各身体部位样本分布')
    plt.xticks(x, body_parts, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(args.output_dir, 'body_part_distribution.png'), dpi=300)
    
    # 图2: 每个身体部位的正负类比例
    plt.figure(figsize=(15, 10))
    
    # 合并数据集以计算每个部位的正负类比例
    df_combined = pd.concat([
        df_train.assign(dataset='训练集'),
        df_valid.assign(dataset='验证集')
    ])
    
    # 使用seaborn进行绘制
    sns.countplot(data=df_combined, x='body_part', hue='label', col='dataset', col_wrap=1)
    plt.title('各身体部位正负类分布')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(args.output_dir, 'class_distribution.png'), dpi=300)
    
    # 图3: 特征分布可视化
    plt.figure(figsize=(15, 15))
    
    feature_cols = ['mean', 'std', 'texture']
    
    for i, feature in enumerate(feature_cols):
        plt.subplot(3, 1, i+1)
        sns.kdeplot(data=df_train, x=feature, hue='body_part', common_norm=False)
        plt.title(f'训练集 {feature} 分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'feature_distribution.png'), dpi=300)
    
    logger.info(f"已保存分布可视化图表到目录: {args.output_dir}")

def print_distribution(df, name):
    """打印分布报告"""
    logger.info(f"\n📊 {name} 数据分布:")
    
    # 总体分布
    logger.info(f"总样本数: {len(df)}")
    logger.info(f"正负类分布: \n{df['label'].value_counts()}")
    
    # 身体部位分布
    body_part_dist = df.groupby('body_part')['label'].count()
    logger.info(f"身体部位分布: \n{body_part_dist}")
    
    # 身体部位和标签的详细分布
    detailed_dist = df.groupby(['body_part', 'label']).size().unstack(fill_value=0)
    detailed_dist.columns = ['正常 (0)', '异常 (1)']
    logger.info(f"详细分布: \n{detailed_dist}")
    
    # 正负类比例
    detailed_dist['比例(异常/总数)'] = detailed_dist['异常 (1)'] / (detailed_dist['正常 (0)'] + detailed_dist['异常 (1)'])
    logger.info(f"各部位异常样本比例: \n{detailed_dist['比例(异常/总数)']}")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    args = setup_environment(args)
    
    logger.info(f"开始处理MURA数据集, 数据根目录: {args.data_root}")
    
    # 获取图像转换函数
    transform = get_transform(args.img_size)
    
    # 处理训练集和验证集
    logger.info("获取训练集统计信息...")
    train_stats = get_patient_stats(args.train_csv, args.data_root, transform, "train")
    
    logger.info("获取验证集统计信息...")
    valid_stats = get_patient_stats(args.valid_csv, args.data_root, transform, "valid")
    
    # 合并数据集
    df_all = pd.concat([train_stats, valid_stats], ignore_index=True)
    if df_all.empty:
        logger.error("处理后的数据为空，请检查输入文件和路径")
        return
    
    logger.info(f"合并后的数据集大小: {df_all.shape}")
    
    # 分层聚类分割
    df_train_final, df_valid_final = stratified_cluster_split(df_all, args)
    
    # 打印分布报告
    print_distribution(df_train_final, "训练集")
    print_distribution(df_valid_final, "验证集")
    
    # 保存结果
    save_results(df_train_final, df_valid_final, args)
    
    # 可视化分布
    visualize_distribution(df_train_final, df_valid_final, args)
    
    logger.info("✅ 完成! 新的训练/验证集CSV已保存.")

if __name__ == "__main__":
    # 命令行示例
    # python mura_balanced_split.py \
    #     --data_root "C:\Users\Vivo\2025_medicalimage_and_AI" \
    #     --train_csv "C:\Users\Vivo\2025_medicalimage_and_AI\MURA-v1.1\train_labeled_studies.csv" \
    #     --valid_csv "C:\Users\Vivo\2025_medicalimage_and_AI\MURA-v1.1\valid_labeled_studies.csv" \
    #     --output_dir "mura_balanced_split" \
    #     --valid_ratio 0.2 \
    #     --patient_level
    print("命令行示例：")
    main()
    
    #python secversion_split.py --data_root "C:\Users\Vivo\2025_medicalimage_and_AI" --train_csv "C:\Users\Vivo\2025_medicalimage_and_AI\MURA-v1.1\train_labeled_studies.csv" --valid_csv "C:\Users\Vivo\2025_medicalimage_and_AI\MURA-v1.1\valid_labeled_studies.csv" --output_dir "C:\Users\Vivo\2025_medicalimage_and_AI\mura_balanced_split" --valid_ratio 0.2 --patient_level