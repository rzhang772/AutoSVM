import os
import pandas as pd

# 文件夹路径
log_folder = './threadlog'
output_file = './threadlog/experiment_summary_values_threads.csv'

# 定义需要提取的统计项名称
configuration_keys = [
    "Task type", "SVM implementation", "Parallel training", "Clustering enabled",
    "Cascade clustering", "Fixed k", "Algorithm", "Method", "k range", "Parallel clustering",
    "entorpy selection", "Entropy ratio",
    "Cluster jobs", 
    # "Sampling enabled", "Sampling ratio", 
    "Data balancing enabled", "Minimum class ratio", "Maximum balance samples", 
    "Feature processing enabled",
    # "Independence check", "Independence threshold", "Independence smoothing",
    "Feature construction", "Mutual info selection",
    # "QBSOFS selection", 
    "Tuning enabled",
    "Optimizer", "Number of iterations", "Cross-validation folds", "Parallel tuning",
    "Tuning jobs",
    "cluster number",
]

results_keys = [
    "Overall Accuracy", "Overall F1 Score", 
    # "Y_range", "Overall MSE", "Overall RMSE", "Overall R² Score"
]

timing_keys = [
    "Loading training data", "Loading test data", "Data normalization", "Clustering",
    "Data balancing", "Feature processing", "SVM processing", "Hyperparameter tuning",
    "Training time", "Testing time", "Prediction time", "Total execution time"
]

# 提取匹配项的方法
def extract_matching_items(log_content, keys, timing = False):
    extracted_data = {key: None for key in keys} 
    for line in log_content.split("\n"):
        for key in keys:
            if key in line:
                parts = line.split(key+":")
                if len(parts) >= 2:
                    value = parts[-1].strip()
                    if timing:
                        value = value.split(' ')[0]
                    extracted_data[key] = [value]
                break
    return extracted_data

# 解析日志文件
def parse_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()
        # print(log_content)
    # 提取配置信息、结果和时间汇总
    config_data = extract_matching_items(log_content, configuration_keys)
    results_data = extract_matching_items(log_content, results_keys)
    timing_data = extract_matching_items(log_content, timing_keys, timing=True)

    # 合并所有数据
    combined_data = {**config_data, **results_data, **timing_data}
    return combined_data

# 主函数
if __name__ == "__main__":
    if not os.path.exists(log_folder):
        print(f"Log folder '{log_folder}' does not exist.")
        exit()

    # 初始化 DataFrame 列表
    df = pd.DataFrame(columns=configuration_keys + results_keys + timing_keys)

    # 遍历所有日志文件
    for log_file in os.listdir(log_folder):
        if log_file.startswith('log_') and log_file.endswith('.log'):
            print(f"Processing log file: {log_file}")
            log_path = os.path.join(log_folder, log_file)
            log_values = parse_log_file(log_path)
            log_values["Log File"] = [log_file]
            log_values['Dataset'] = [log_file.split('_')[1]]
            log_df = pd.DataFrame(log_values)
            df = pd.concat([df, log_df], axis=0, ignore_index=True)

    # 保存为 CSV 文件
    df = df[['Dataset'] + results_keys + timing_keys + configuration_keys + ['Log File']]
    sorted_df = df.sort_values(by=['Task type','Log File'])
    sorted_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Experiment summary saved to {output_file}.")