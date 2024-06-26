"""123
"""

def extract_keyword(raw_doc):
    """123
    """
    print("enter extract_keyword")

    first_line = raw_doc.split("\n")[0]
    first_line = first_line.split("：")[0]
    first_line = first_line.split("(")[0]
    first_line = first_line.split("（")[0]
    split_first_line = first_line.split(" ")[:3]
    print(f"split_first_line = {split_first_line}")
    length = len(split_first_line)
    print(f"length = {length}")


    keyword_type = 'Pathogenic'
    if length==3:
        class_name, type_name, keyword = split_first_line

        if keyword=='其他害蟲的基本資訊':
            keyword = type_name
            keyword_type = 'crop'
        elif keyword=='貯倉管理與防治對策':
            keyword = type_name
        elif keyword=='跟蟎類':
            keyword = '根蟎類'

    elif length==2:
        class_name, keyword = split_first_line

    elif length==1:

        if len(split_first_line[0])>20:
            keyword = "白絹病"
        else:
            keyword = split_first_line[0]

    if "的基本資訊" in keyword or "的防治方法" in keyword:
        keyword = keyword[:-6]

    print(f"\tkeyword = {keyword}")
    print("exit extract_keyword")

    return keyword, keyword_type



# def get_prompt(title: List[str], document: str) -> str:
#     """根據提供的標題和文檔內容生成提示信息。"""
#     prompt = '你現在是一位農業病蟲害防治專家，你將會看到一份markdown格式的表格，可能為害蟲基本資訊或是害蟲防治方法，請根據此參考文章生成一組問題及答案，問題必須包含表格內的生物學名，問題內容可包括防治方法、藥劑名稱、每公頃使用量、稀釋倍數、施藥方法、注意事項等等，若文章不包含以上資訊也可以生成其他類型問題，最後將這些問題及答案以範例格式儲存'+'[{"question":"q","answer":"a"}]\n\n'\
#     + f"類別：{title[0]}\t作物名稱：{title[1]}\t害蟲名稱(基本資訊/防治方法)：{title[2]}\n\n"\
#     + f"參考文章：\n{document}"

#     return prompt
