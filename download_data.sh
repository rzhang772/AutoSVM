#!/bin/bash

urls=(
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.tr.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic.t.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic_scale.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic_scale.t.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/seismic.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/seismic.t.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/seismic_scale.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/seismic_scale.t.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/smallNORB.xz" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/smallNORB.t.xz" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/smallNORB-32x32.xz" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/smallNORB-32x32.t.xz" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/tmc2007_train.svm.bz2" 
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/tmc2007_test.svm.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/criteo.kaggle2014.svm.tar.xz"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined_normalized.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2"
    # "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale"
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2"
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.tr.bz2"
)

# prefix_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
# class = "multiclass"
# dataset_name = ()

# Loop through each URL in the array
for url in "${urls[@]}"; do
    echo "Processing URL: $url"
    # Use wget to download the content
    wget -P ./data/ "$url"
    
    # Check if wget was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $url"
    else
        echo "Failed to download $url"
    fi
done