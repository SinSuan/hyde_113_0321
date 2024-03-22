"""123
"""

import json
import subprocess
import numpy as np
import faiss
from pyserini.search.lucene import LuceneSearcher
from FlagEmbedding import BGEM3FlagModel

# def files_2_index(path_2_dir):
#     """formatter
#     """
#     print("enter files_2_index")



#     print("exit files_2_index")
#     return index_json


def corpus_build(retriever_type, path_2_raw_data, path_2_corpus):
    """

    Parameters
    ----------
    retriever_type : str
        'BM25' or 'BGEM3'

    Returns
    -------
    None
    """
    print("enter corpus_build")

    if retriever_type=='BM25':
        print("\tenter if 'BM25'")

        command = [
            'python', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', path_2_raw_data,
            '--language', 'zh',
            '--index', path_2_corpus,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', '1',
            '--storePositions', '--storeDocvectors', '--storeRaw'
        ]

        # 執行命令
        subprocess.run(command, check=False)
        # The ``check`` keyword is set to False by default. It means the process launched by ``subprocess.run`` can exit with a non-zero exit code and fail silently. It's better to set it explicitly to make clear what the error-handling behavior is.

        print("\texit if 'BM25'")

    elif retriever_type=='BGEM3':
        print("\tenter elif 'BGEM3'")

        # prepocess ttl_docs
        with open(path_2_raw_data, 'r', encoding='utf-8') as file:
            total_data = json.load(file)
        ttl_doc = [data['contents'] for data in total_data]

        # create corpus
        encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        corpus = encoder.encode(
            ttl_doc,
            batch_size=12,
            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
            )['dense_vecs']
        corpus = corpus.astype(np.float32)

        # save corpus
        np.save(path_2_corpus, corpus)

        print("\texit elif 'BGEM3'")

    print("exit corpus_build")

def init_retriever(retriever_type, path_2_corpus):
    """123
    """
    print("enter init_retriever")

    if retriever_type=='BM25':
        print("\tenter if 'BM25'")

        # init BM25
        # connect BM25 to corpus
        retriever = LuceneSearcher(path_2_corpus)
        retriever.set_language('zh')

        print("\texit if 'BM25'")

    elif retriever_type=='BGEM3':
        print("\tenter elif 'BGEM3'")

        # init faiss
        d = 1024
        retriever = faiss.IndexFlatL2(d)

        # connect faiss to corpus
        corpus = np.load(path_2_corpus)
        retriever.add(corpus)   # 這行沒有 bug，跟官方文件寫法一樣，且跑起來是正確的

        print("\texit elif 'BGEM3'")

    retriever_info = [retriever_type, retriever]

    print("exit init_retriever")
    return retriever_info

def apply_retriever(retriever_info, ttl_query, top_k=15):
    r"""

    Parameters
    ----------
    query : List[str]
        total queries

    Returns
    -------
    List[List[int]]
        retrieved docid(s) for each query
    """
    print("enter apply_retriever")

    retriever_type, retriever = retriever_info

    if retriever_type=='BM25':
        print("\tenter if 'BM25'")

        ttl_hits = []
        for query in ttl_query:
            hits = retriever.search(query, top_k)
            hits = [ hit.docid for hit in hits ]
            ttl_hits.append(hits)

        print("\texit if 'BM25'")

    elif retriever_type=='BGEM3':
        print("\tenter elif 'BGEM3'")

        encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        query_vec = encoder.encode(
            ttl_query,
            batch_size=12,
            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
            )['dense_vecs']

        _, ttl_hits = retriever.search(query_vec, top_k)

        print("\texit elif 'BGEM3'")

    print("exit apply_retriever")
    return ttl_hits

# if __name__ == "__main__":
#     # retriever_type = 'BM25'
#     # path_2_raw_data = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/raw_data'
#     # path_2_corpus = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/BM25'

#     retriever_type = 'BGEM3'
#     path_2_raw_data = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/raw_data/raw_data.json'
#     path_2_corpus = '/user_data/DG/hyde_113_0321/garbage_code/test_retriver/BGEM3/113_0322_1917.npy'

#     ttl_query = ['松鼠吃葡萄嗎?']
#     top_k = 15
    
#     corpus_build(retriever_type, path_2_raw_data, path_2_corpus)
#     retriever_info = init_retriever(retriever_type, path_2_corpus)
#     ttl_hits = apply_retriever(retriever_info, ttl_query, top_k=15)
    
#     print(f"ttl_hits = {ttl_hits}")