source /conda_env/orz.sh
conda activate retriever

save_path=/cpfs01/user/zhoujiecheng/workload_rl_analyse/search-r1/local_dense_retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/cpfs01/user/zhoujiecheng/workload_rl_analyse/models/e5-base-v2

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu