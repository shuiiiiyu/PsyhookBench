import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def batch_evaluate_one_col_flat(one_col_dir, model_results_folder):

    if not os.path.exists(one_col_dir) or not os.path.exists(model_results_folder):
        print("错误：请检查输入文件夹路径。")
        return
    model_files = [f for f in os.listdir(model_results_folder) if f.endswith('.csv')]
    gt_files = [f for f in os.listdir(one_col_dir) if f.endswith('.csv')]

    all_rows = []

    for m_file in model_files:
        model_name = os.path.splitext(m_file)[0]
        model_path = os.path.join(model_results_folder, m_file)
        
        try:
            df_all_pred = pd.read_csv(model_path)
            df_all_pred['post_id'] = df_all_pred['post_id'].astype(str).str.strip()
            df_all_pred.columns = df_all_pred.columns.astype(str)
            
            print(f"正在处理模型: {model_name}")

            for g_file in gt_files:
                mechanism_id = "".join(filter(str.isdigit, g_file))
                col_pred = f"h{mechanism_id}"
                
                if col_pred not in df_all_pred.columns:
                    continue
                    
                file_path = os.path.join(one_col_dir, g_file)
                df_gt = pd.read_csv(file_path)
                df_gt['post_id'] = df_gt['post_id'].astype(str).str.strip()
                gt_post_ids = set(df_gt['post_id'].unique())

                df_eval = df_all_pred[['post_id', col_pred]].copy()
                df_eval['y_true'] = df_eval['post_id'].apply(lambda x: 1 if x in gt_post_ids else 0)
                
                y_pred = df_eval[col_pred].fillna(0).astype(int)
                y_true = df_eval['y_true'].astype(int)

                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()

                all_rows.append({
                    'Model': model_name,
                    'Mechanism': col_pred,
                    'GT_File': g_file,
                    'Match_Count': len(df_eval),
                    'TP': tp, 
                    'FP': fp, 
                    'FN': fn,
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'F1_Score': round(f1, 4)
                })
        except Exception as e:
            print(f"读取模型 {m_file} 出错: {e}")

    if all_rows:
        df_final = pd.DataFrame(all_rows)
        df_final = df_final.sort_values(by=['Model', 'Mechanism'])
        
        output_name = "HOOKS.csv"
        df_final.to_csv(output_name, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*70)
        print(f"✅ 处理完成！已将所有组合结果写入大表。")
        print(f"文件路径: {output_name}")
        print("="*70)
        print(df_final.head(10).to_string(index=False))
    else:
        print("❌ 未生成任何有效数据。")

if __name__ == "__main__":
    ONE_COL_DIR = 'check_csvs_fewshot_/one_col'
    MODELS_DIR = '测试结果'
    
    batch_evaluate_one_col_flat(ONE_COL_DIR, MODELS_DIR)