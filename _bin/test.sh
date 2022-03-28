#!/bin/sh

cd `dirname $0`
cd ..

INPUT="electronics"
OUTPUT="_out/electronics"

categorical_idxs="brand/category_code"
# categorical_idxs="brand/product_id"

tag="test"
alpha=1
beta=1
N_ITER=30


python3 _src/main.py --input_tag $INPUT \
                                  --out_dir $OUTPUT"/infinite_unigram/"$tag \
                                  --categorical_idxs "$categorical_idxs" \
                                  --verbose \
                                  --alpha 1 \
                                  --beta 1 \
                                  --N_ITER 30 