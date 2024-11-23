#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run test and check result
run_test() {
    echo -e "\n${GREEN}Running test: $1${NC}"
    echo "Command: $2"
    eval $2
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Test passed${NC}"
    else
        echo -e "${RED}Test failed${NC}"
    fi
}

# Test datasets
CLF_TRAIN="./processed/clf/aloi_train.txt"
CLF_TEST="./processed/clf/aloi_test.txt"
REG_TRAIN="./processed/reg/cadata_train.txt"
REG_TEST="./processed/reg/cadata_test.txt"

echo "=== Starting AutoSVM Tests ==="



# # 1. Classification Tests with kmeans
# echo -e "\n=== Classification Tests with kmeans ==="

# # Silhouette method
# run_test "KMeans with Silhouette (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm kmeans --method silhouette \
#     --svm-type libsvm"

# # GAP method
# run_test "KMeans with GAP (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm kmeans --method gap \
#     --svm-type libsvm"

# # 2. Classification Tests with fixed k
# echo -e "\n=== Classification Tests with fixed k ==="

# # Random clustering
# run_test "Random Clustering (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm random --k 5 \
#     --svm-type libsvm"

# # FIFO clustering
# run_test "FIFO Clustering (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm fifo --k 5 \
#     --svm-type libsvm"

# # 3. Feature Processing Tests (Classification)
# echo -e "\n=== Feature Processing Tests (Classification) ==="

# # All feature processing options
# run_test "KMeans with All Feature Processing (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm kmeans --method silhouette \
#     --feature-processing --feature-construction --chi2 --qbsofs \
#     --svm-type libsvm"

# # Only chi-square
# run_test "KMeans with Chi-square (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm kmeans --method silhouette \
#     --feature-processing --chi2 \
#     --svm-type libsvm"

# # 4. Regression Tests
# echo -e "\n=== Regression Tests ==="

# # Basic regression with kmeans
# run_test "KMeans (REG)" "python src/main.py \
#     --train $REG_TRAIN --test $REG_TEST \
#     --type reg --algorithm kmeans --method silhouette \
#     --svm-type libsvm"

# # Regression with feature processing
# run_test "KMeans with Feature Processing (REG)" "python src/main.py \
#     --train $REG_TRAIN --test $REG_TEST \
#     --type reg --algorithm kmeans --method silhouette \
#     --feature-processing --feature-construction --chi2 --qbsofs \
#     --svm-type libsvm"

# # 5. ThunderSVM Tests
# echo -e "\n=== ThunderSVM Tests ==="

# # Classification with ThunderSVM
# run_test "ThunderSVM (CLF)" "python src/main.py \
#     --train $CLF_TRAIN --test $CLF_TEST \
#     --type clf --algorithm kmeans --method silhouette \
#     --svm-type thundersvm"

# # Regression with ThunderSVM
# run_test "ThunderSVM (REG)" "python src/main.py \
#     --train $REG_TRAIN --test $REG_TEST \
#     --type reg --algorithm kmeans --method silhouette \
#     --svm-type thundersvm"

# echo -e "\n=== All Tests Completed ===" 