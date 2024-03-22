from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    return f"敝處有幸受託轉換文言文與白話文，望能即時且準確完成此一任務。USER: {instruction} ASSISTANT:" # zero2
    # return f"請根據接下來的指令，將以下文字轉換文言文與白話文。USER: {instruction} ASSISTANT:"
    # return f"USER: {instruction} ASSISTANT:"
    # return f"敝處有幸受託轉換文言文與白話文，望能即時且準確完成此一任務。 \
    #         例如： \
    #         Input: 翻譯成文言文：於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。\
    #         Output: 帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。\
    #         Input: 文言文翻譯：\n靈鑒忽臨，忻歡交集，乃迴燈拂席以延之。\
    #         Output: 答案：靈仙忽然光臨，趙旭歡欣交集，於是他就把燈點亮，拂拭乾淨床席來延請仙女。\
    #         USER: {instruction} ASSISTANT:"
    # return f"敝處有幸受託轉換文言文與白話文，望能即時且準確完成此一任務。\
    # 例如：\
    # Input: 翻譯成文言文：於是，廢帝讓瀋慶之的堂侄、直將軍瀋攸之賜瀋慶之毒藥，命瀋慶之自殺。\
    # Output: 帝乃使慶之從父兄子直閣將軍攸之賜慶之藥。\
    # Input: 文言文翻譯：\n靈鑒忽臨，忻歡交集，乃迴燈拂席以延之。\
    # Output: 答案：靈仙忽然光臨，趙旭歡欣交集，於是他就把燈點亮，拂拭乾淨床席來延請仙女。\
    # Iuput: 文言文翻譯：\n契丹主以陽城之戰為彥卿所敗，詰之。彥卿曰： 臣當時惟知為晉主竭力，今日死生惟命。\
    # Output: 答案：契丹主因陽城之戰被符彥卿打敗，追問符彥卿，彥卿說： 臣當時隻知為晉主竭盡全力，今日死生聽你決定。\
    # Input: 世祖說： 你是認為夢炎比葉李要賢嗎？\n這句話在中國古代怎麼說：\
    # Output: 帝曰： 汝以夢炎賢於李耶？\
    # USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    # compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,    # True?
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return bnb_config
