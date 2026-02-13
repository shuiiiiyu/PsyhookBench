import os
import pandas as pd
from sklearn.metrics import f1_score

def batch_evaluate_f1_pivot(one_col_dir, model_results_folder):
    if not os.path.exists(one_col_dir) or not os.path.exists(model_results_folder):
        print("错误：请检查输入文件夹路径。")
        return

    model_files = [f for f in os.listdir(model_results_folder) if f.endswith('.csv')]
    gt_files = [f for f in os.listdir(one_col_dir) if f.endswith('.csv')]
    
    all_f1_results = []

    for m_file in model_files:
        model_name = os.path.splitext(m_file)[0]
        model_path = os.path.join(model_results_folder, m_file)
        
        try:
            df_all_pred = pd.read_csv(model_path)
            df_all_pred['post_id'] = df_all_pred['post_id'].astype(str).str.strip()
            df_all_pred.columns = df_all_pred.columns.astype(str)
            
            print(f"正在评估模型: {model_name}")

            for g_file in gt_files:
                mechanism_id = "".join(filter(str.isdigit, g_file))
                col_pred = f"h{mechanism_id}"
                
                if col_pred not in df_all_pred.columns:
                    continue
                    
                file_path = os.path.join(one_col_dir, g_file)
                df_gt = pd.read_csv(file_path)
                df_gt['post_id'] = df_gt['post_id'].astype(str).str.strip()
                gt_post_ids = set(df_gt['post_id'].unique())

                y_true = df_all_pred['post_id'].apply(lambda x: 1 if x in gt_post_ids else 0)
                y_pred = df_all_pred[col_pred].fillna(0).astype(int)
                
                f1 = f1_score(y_true, y_pred, zero_division=0)

                all_f1_results.append({
                    'Model': model_name,
                    'Mechanism': mechanism_id,
                    'F1': round(f1, 4)
                })
        except Exception as e:
            print(f"处理模型 {m_file} 出错: {e}")

    if all_f1_results:
        df_results = pd.DataFrame(all_f1_results)
        pivot_df = df_results.pivot(index='Model', columns='Mechanism', values='F1')
        pivot_df.columns = sorted(pivot_df.columns, key=int)
        pivot_df['Average'] = pivot_df.mean(axis=1).round(4)
        pivot_df = pivot_df.sort_values(by='Average', ascending=False)
        
        output_name = "MODEL_F1_COMPARISON.csv"
        pivot_df.to_csv(output_name, encoding='utf-8-sig')
        
        print("\n" + "="*70)
        print(f"✅ 评估完成！F1 对比总表已生成。")
        print(f"保存路径: {output_name}")
        print("="*70)
        print(pivot_df.to_string())
    else:
        print("❌ 未生成任何数据。")

if __name__ == "__main__":
    ONE_COL_DIR = 'check/one_col'
    MODELS_DIR = '测试结果'
    
    batch_evaluate_f1_pivot(ONE_COL_DIR, MODELS_DIR)