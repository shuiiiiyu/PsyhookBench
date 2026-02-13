import os
import pandas as pd
from sklearn.metrics import f1_score

def run_batch_evaluation_pivot(check_status_folder, model_results_folder):
    master_summary = []
    
    if not os.path.exists(model_results_folder):
        print(f"错误：找不到目录 {model_results_folder}")
        return

    model_files = [f for f in os.listdir(model_results_folder) if f.endswith('.csv')]
    
    # 定义标签列
    h_cols = [f'h{i}' for i in range(1, 9)]
    pred_cols = [str(i) for i in range(1, 9)]

    # 1. 遍历每一个模型
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(model_results_folder, model_file)
        
        try:
            df_compare = pd.read_csv(model_path)
            # 强行清洗 ID 防止匹配失败
            df_compare['post_id'] = df_compare['post_id'].astype(str).str.strip()
            
            print(f"正在处理模型: {model_name}")

            # 2. 遍历每一个基准文件
            for root, dirs, files in os.walk(check_status_folder):
                for file in files:
                    if file.endswith('.csv'):
                        # 去掉 .csv 后缀用于展示
                        display_name = file.replace('.csv', '')
                        
                        file_path = os.path.join(root, file)
                        df_check = pd.read_csv(file_path)
                        df_check['post_id'] = df_check['post_id'].astype(str).str.strip()
                        
                        # 合并
                        merged = pd.merge(df_check, df_compare, on='post_id', suffixes=('_true', '_pred'))
                        
                        if merged.empty:
                            continue

                        # 二值化数值提取
                        y_true = (merged[h_cols].fillna(0).values != 0).astype(int)
                        y_pred = (merged[pred_cols].fillna(0).values != 0).astype(int)

                        # 计算 Macro-F1
                        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                        
                        master_summary.append({
                            'Model': model_name,
                            'BaseFile': display_name,
                            'F1': round(macro_f1, 4)
                        })
        except Exception as e:
            print(f"读取模型 {model_file} 出错: {e}")

    # 3. 转换为透视表格式
    if master_summary:
        df_flat = pd.DataFrame(master_summary)
        # 旋转：行是模型，列是基准文件
        df_pivot = df_flat.pivot(index='Model', columns='BaseFile', values='F1')
        
        # 计算模型平均分并降序排列
        df_pivot['Average'] = df_pivot.mean(axis=1).round(4)
        df_pivot = df_pivot.sort_values(by='Average', ascending=False)
        
        output_name = "CLASS_PIVOT.csv"
        df_pivot.to_csv(output_name, encoding='utf-8-sig')
        
        print("\n" + "="*80)
        print(f"✅ 处理完成！透视汇总表已生成。")
        print(f"保存路径: {output_name}")
        print("="*80)
        print(df_pivot.to_string())
    else:
        print("❌ 未生成任何有效数据。")

if __name__ == "__main__":
    # check/class: 基准文件夹
    # 测试结果: 模型文件夹
    run_batch_evaluation_pivot('check/class', '测试结果')