import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

def calculate_metrics(y_true, y_pred):
    """计算多标签分类的核心指标"""
    # 1. EMR
    emr = accuracy_score(y_true, y_pred)
    # 2. 汉明损失
    h_loss = hamming_loss(y_true, y_pred)
    # 3. Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # 4. Micro-F1
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return {
        'EMR': round(emr, 4),
        'HammingLoss': round(h_loss, 4),
        'Macro-F1': round(macro_f1, 4),
        'Micro-F1': round(micro_f1, 4)
    }

def run_batch_evaluation_all_pairs(check_status_folder, model_results_folder):
    master_summary = []
    if not os.path.exists(model_results_folder):
        print(f"错误：找不到目录 {model_results_folder}")
        return

    model_files = [f for f in os.listdir(model_results_folder) if f.endswith('.csv')]
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(model_results_folder, model_file)
        df_compare = pd.read_csv(model_path)

        h_cols = [f'h{i}' for i in range(1, 9)]
        pred_cols = [str(i) for i in range(1, 9)]
        
        print(f"正在处理模型: {model_name} ...")

        for root, dirs, files in os.walk(check_status_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df_check = pd.read_csv(file_path)
                    
                    merged = pd.merge(df_check, df_compare, on='post_id', suffixes=('_pred', '_true'))
                    
                    if merged.empty:
                        continue

                    y_pred = (merged[pred_cols].fillna(0).values != 0).astype(int)
                    y_true = (merged[h_cols].fillna(0).values != 0).astype(int)

                    res = calculate_metrics(y_true, y_pred)

                    master_summary.append({
                        'ModelName': model_name,
                        'TestFile': file,
                        'MatchCount': 3041,
                        'Macro-F1': res['Macro-F1'],
                        'Micro-F1': res['Micro-F1'],
                        'HammingLoss': res['HammingLoss'],
                        'EMR': res['EMR']
                    })

    if master_summary:
        df_master = pd.DataFrame(master_summary)
        df_master = df_master.sort_values(by=['ModelName', 'TestFile'])
        
        output_name = "全样本-zeroshot.csv"
        df_master.to_csv(output_name, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*80)
        print(f"✅ 评估完成！汇总表包含所有模型与文件的对比。")
        print(f"文件保存至: {output_name}")
        print("="*80)
        print(df_master.head(20).to_string(index=False))
    else:
        print("❌ 未生成任何有效数据，请检查文件 post_id 是否匹配或路径是否正确。")

if __name__ == "__main__":
    run_batch_evaluation_all_pairs('check_csvs_zeroshot', 'raw_data_results_zeroshot')