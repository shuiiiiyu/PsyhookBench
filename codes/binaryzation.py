import pandas as pd
import os

def binarize_csv_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在")
        return

    target_columns = [str(i) for i in range(1, 9)] # ['1', '2', ..., '8']

    print(f"开始处理文件夹: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            
            try:
                df = pd.read_csv(file_path)
                existing_cols = [col for col in target_columns if col in df.columns]
                
                if not existing_cols:
                    print(f"跳过 {file}: 未找到列 1-8")
                    continue
                for col in existing_cols:
                    # 使用 pd.to_numeric 处理可能存在的非数值字符，强转为 0
                    nums = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    # 只要不等于 0，就记为 1
                    df[col] = (nums != 0).astype(int)

                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"已处理并覆盖: {file}")

            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")

    print("\n所有符合条件的 CSV 文件已处理完毕。")

if __name__ == "__main__":
    folder_to_process = 'classified_output' 
    binarize_csv_files(folder_to_process)