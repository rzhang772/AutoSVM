import os
import subprocess
from datetime import datetime

# 数据集路径列表及类型
DATASETS = [
    #clf
    # { "entropy": "0.1", "bestk": "98", "train": "./data/processed/clf/aloi_train.txt", "test": "./data/processed/clf/aloi_test.txt", "type": "clf",},
    # { "entropy": "0.3", "bestk": "5", "train": "./data/processed/clf/aloi_train.txt", "test": "./data/processed/clf/aloi_test.txt", "type": "clf",},
    # { "entropy": "0.5", "bestk": "3", "train": "./data/processed/clf/aloi_train.txt", "test": "./data/processed/clf/aloi_test.txt", "type": "clf",},
    # { "entropy": "0.8", "bestk": "4", "train": "./data/processed/clf/aloi_train.txt", "test": "./data/processed/clf/aloi_test.txt", "type": "clf",},
    # { "entropy": "1.0", "bestk": "2", "train": "./data/processed/clf/aloi_train.txt", "test": "./data/processed/clf/aloi_test.txt", "type": "clf",},

    
    # {"entropy": "0.1","bestk": "2", "train": "./data/processed/clf/news20_train.txt", "test": "./data/processed/clf/news20_test.txt", "type": "clf"},
    # {"entropy": "0.3","bestk": "4", "train": "./data/processed/clf/news20_train.txt", "test": "./data/processed/clf/news20_test.txt", "type": "clf"},
    # {"entropy": "0.5","bestk": "5", "train": "./data/processed/clf/news20_train.txt", "test": "./data/processed/clf/news20_test.txt", "type": "clf"},
    # {"entropy": "0.8","bestk": "3", "train": "./data/processed/clf/news20_train.txt", "test": "./data/processed/clf/news20_test.txt", "type": "clf"},
    # {"entropy": "1.0","bestk": "2", "train": "./data/processed/clf/news20_train.txt", "test": "./data/processed/clf/news20_test.txt", "type": "clf"},
    
    
    # {"entropy": "0.1","bestk": "2", "train": "./data/processed/clf/sector.scale", "test": "./data/processed/clf/sector.t.scale", "type": "clf"},
    # {"entropy": "0.3","bestk": "2", "train": "./data/processed/clf/sector.scale", "test": "./data/processed/clf/sector.t.scale", "type": "clf"},
    # {"entropy": "0.5","bestk": "3", "train": "./data/processed/clf/sector.scale", "test": "./data/processed/clf/sector.t.scale", "type": "clf"},
    # {"entropy": "0.8","bestk": "2", "train": "./data/processed/clf/sector.scale", "test": "./data/processed/clf/sector.t.scale", "type": "clf"},
    # {"entropy": "1.0","bestk": "41", "train": "./data/processed/clf/sector.scale", "test": "./data/processed/clf/sector.t.scale", "type": "clf"},



    # {"entropy": "0.1","bestk": "28", "train": "./data/processed/clf/avazu-app.tr", "test": "./data/processed/clf/avazu-app.t", "type": "clf"},
    {"entropy": "0.3","bestk": "32", "train": "./data/processed/clf/avazu-app.tr", "test": "./data/processed/clf/avazu-app.t", "type": "clf"},
    # {"entropy": "0.5","bestk": "100", "train": "./data/processed/clf/avazu-app.tr", "test": "./data/processed/clf/avazu-app.t", "type": "clf"},
    # {"entropy": "0.8","bestk": "", "train": "./data/processed/clf/avazu-app.tr", "test": "./data/processed/clf/avazu-app.t", "type": "clf"},
    # {"entropy": "1.0","bestk": "19", "train": "./data/processed/clf/avazu-app.tr", "test": "./data/processed/clf/avazu-app.t", "type": "clf"},

    #   {"entropy": "0.1","bestk": "35", "train": "./data/processed/clf/url_combined_normalized_train.txt", "test": "./data/processed/clf/url_combined_normalized_test.txt", "type": "clf"},
    # {"entropy": "0.3","bestk": "97", "train": "./data/processed/clf/url_combined_normalized_train.txt", "test": "./data/processed/clf/url_combined_normalized_test.txt", "type": "clf"},
    # {"entropy": "0.5","bestk": "100", "train": "./data/processed/clf/url_combined_normalized_train.txt", "test": "./data/processed/clf/url_combined_normalized_test.txt", "type": "clf"},
    # {"entropy": "0.8","bestk": "3", "train": "./data/processed/clf/url_combined_normalized_train.txt", "test": "./data/processed/clf/url_combined_normalized_test.txt", "type": "clf"},
    # {"entropy": "1.0","bestk": "2", "train": "./data/processed/clf/url_combined_normalized_train.txt", "test": "./data/processed/clf/url_combined_normalized_test.txt", "type": "clf"},


    # { "entropy": "0.1", "bestk": "2", "train": "./data/processed/clf/HIGGS_train.txt", "test": "./data/processed/clf/HIGGS_test.txt", "type": "clf"},
    # { "entropy": "0.3", "bestk": "10", "train": "./data/processed/clf/HIGGS_train.txt", "test": "./data/processed/clf/HIGGS_test.txt", "type": "clf"},
    # { "entropy": "0.5", "bestk": "6", "train": "./data/processed/clf/HIGGS_train.txt", "test": "./data/processed/clf/HIGGS_test.txt", "type": "clf"},
    # { "entropy": "0.8", "bestk": "3", "train": "./data/processed/clf/HIGGS_train.txt", "test": "./data/processed/clf/HIGGS_test.txt", "type": "clf"},
    # { "entropy": "1.0", "bestk": "4", "train": "./data/processed/clf/HIGGS_train.txt", "test": "./data/processed/clf/HIGGS_test.txt", "type": "clf"},

    # {"entropy": "0.1","bestk": "", "train": "./data/processed/clf/epsilon_normalized", "test": "./data/processed/clf/epsilon_normalized.t", "type": "clf"},
    # {"entropy": "0.3","bestk": "9", "train": "./data/processed/clf/epsilon_normalized", "test": "./data/processed/clf/epsilon_normalized.t", "type": "clf"},
    # {"entropy": "0.5","bestk": "", "train": "./data/processed/clf/epsilon_normalized", "test": "./data/processed/clf/epsilon_normalized.t", "type": "clf"},
    # {"entropy": "0.8","bestk": "", "train": "./data/processed/clf/epsilon_normalized", "test": "./data/processed/clf/epsilon_normalized.t", "type": "clf"},
    # {"entropy": "1.0","bestk": "2", "train": "./data/processed/clf/epsilon_normalized", "test": "./data/processed/clf/epsilon_normalized.t", "type": "clf"},

    # # reg
    # {"bestk": "2", "train": "./data/processed/reg/cadata_train.txt", "test": "./data/processed/reg/cadata_test.txt", "type": "reg"},
    # {"bestk": "2", "train": "./data/processed/reg/log1p.E2006.train", "test": "./data/processed/reg/log1p.E2006.test", "type": "reg"},
    # {"bestk": "2", "train": "./data/processed/reg/YearPredictionMSD", "test": "./data/processed/reg/YearPredictionMSD.t", "type": "reg"},
    # {"bestk": "2", "train": "./data/processed/reg/space_ga_scale_train.txt", "test": "./data/processed/reg/space_ga_scale_test.txt", "type": "reg"},
    

]
    # {"bestk": "0", "train": "./data/processed/reg/E2006.train", "test": "./data/processed/reg/E2006.test", "type": "reg"},
# {"bestk": "98", "train": "./data/processed/clf/kdda", "test": "./data/processed/clf/kdda.t", "type": "clf"},

    # {"bestk": "5", "train": "./data/processed/clf/smallNORB-32x32", "test": "./data/processed/clf/smallNORB-32x32.t", "type": "clf"},

ENTROPY_RATIO = [
    # "0.1",
    # "0.3",
    # "0.5",
    # "0.8",
    "1.0"
    ]

# 日志级别
LOG_LEVEL = "debug"

# 获取当前时间
start_time = datetime.now()
print(f"Script started at {start_time}")

# 遍历数据集并运行脚本
for dataset in DATASETS:
    entropy = dataset["entropy"]
    train_path = dataset["train"]
    test_path = dataset["test"]
    dataset_type = dataset["type"]
    bestk = dataset["bestk"]
    
    for eratio in ENTROPY_RATIO:
        # 检查文件是否存在
        if os.path.exists(train_path) and os.path.exists(test_path):
            # 定义基础命令
            command = [
                "python", "src/main.py",
                "--train", train_path,
                "--test", test_path,
                "--type", dataset_type,
                "--parallel-train",

                "--clustering",
                "--cascade",
                "--entropy-selection", "--entropy-ratio", entropy,
                "--k", bestk,
                "--algorithm", "kmeans",
                # "--parallel-cluster",

                "--balance-data", "--min-class-ratio", "0.1",
                
                "--feature-processing",
                "--mutual-info", 
                "--mutual-ratio", "50",
                # "--feature-construction", "--discretize-ratio", "0.1",
                # "--parallel-feature",
                
                # "--tune-hyperparams",
                # "--optimizer", "bayes",
                # "--parallel-tuning",
                "--log-level", LOG_LEVEL
            ]
            
            # 打印并执行命令
            print(f"Running command: {' '.join(command)}")
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 打印输出日志
            print(f"Output:\n{process.stdout}")
            print(f"Errors:\n{process.stderr}")
            
            # 检查返回代码
            if process.returncode != 0:
                print(f"Error occurred while processing dataset: {train_path}, {test_path}")
    else:
        print(f"Dataset files not found: {train_path}, {test_path}")

# 打印完成时间
end_time = datetime.now()
print(f"Script finished at {end_time}. Total time: {end_time - start_time}")

# single svm: tune-hyperparams, optimizer=bayes
# ca-svm: clustering(kmeans(k=256, best),fifo(k=256),average(k=256)), balance, tune-hyperparams, optimizer=bayes
# autosvm: all