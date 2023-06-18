from chains.local_doc_qa import LocalDocQA
import os

if __name__ == "__main__":
    # 初始化消息
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg()
    query = "阿木无人机起飞侧翻，如何解决？"
    vs_path = "/home/amov/langchain/knowledge_base"
    # file = "/home/amov/langchain/dataset/amov.txt"
    # local_doc_qa.get_vector_store(vs_path, files=file, sentence_size=10,  one_conent="阿木无人机起飞侧翻，需要：1.电调信号线与主控接线是否正确对应。2.电机转向和桨叶的安装是否正确。", one_content_segmentation=None)

    #遍历所有文件
    file = list()
    for root, dirs, files_name in os.walk("/home/amov/langchain/dataset"):
        for file_name in files_name:
            file.append(os.path.join(root, file_name))
    local_doc_qa.get_vector_store(vs_path, files=file, sentence_size=10,  one_conent="", one_content_segmentation=None)

    vs_path_answer =os.path.join(vs_path,"vector_store")
    prompt = local_doc_qa.get_knowledge_based_answer(query=query,vs_path=vs_path_answer)
    print(prompt)
    # prompt  =  local_doc_qa.get_search_result_based_answer(query=query)
    # print(prompt)

