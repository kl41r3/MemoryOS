import json
from datetime import datetime, timedelta
from short_term_memory import ShortTermMemory
from mid_term_memory import MidTermMemory
from long_term_memory import LongTermMemory
from dynamic_update import DynamicUpdate
from retrieval_and_answer import RetrievalAndAnswer
from utils import OpenAIClient, gpt_generate_answer, gpt_extract_theme, gpt_update_profile, gpt_generate_multi_summary, get_timestamp, llm_extract_keywords, gpt_personality_analysis
import re
import openai
import time
import tiktoken
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import nltk
total_tokens = 0
num_samples=0

# Initialize OpenAI client
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAIClient(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# Heat threshold
H_THRESHOLD = 5.0

def update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client):
    """
    Update user profile if heat exceeds threshold and extract assistant knowledge.
    """
    if not mid_mem.heap:
        return
    
    neg_heat, sid = mid_mem.heap[0]
    mid_mem.rebuild_heap()
    current_heat = -neg_heat
    
    if current_heat >= H_THRESHOLD:
        session = mid_mem.sessions.get(sid)
        if not session:
            return
        
        un_analyzed = [p for p in session["details"] if not p.get("analyzed", False)]
        if un_analyzed:
            print(f"Updating user profile: Segment {sid} heat {current_heat:.2f} exceeds threshold, starting profile update...")
            
            old_profile = long_mem.get_raw_user_profile(sample_id)
            
            result = gpt_personality_analysis(un_analyzed, client)
            new_profile = result["profile"]
            new_private = result["private"]
            assistant_knowledge = result["assistant_knowledge"]
            
            if old_profile:
                updated_profile = gpt_update_profile(old_profile, new_profile, client)
            else:
                updated_profile = new_profile
                
            long_mem.update_user_profile(sample_id, updated_profile)
            
            # 修改点：拆分 new_private 并逐个存储
            if new_private and new_private != "- None":
                # 按行拆分，过滤空行和非事实行（如 "【User Data】" 或注释）
                facts = [line.strip() for line in new_private.split("\n")]
                for fact in facts:
                    long_mem.add_knowledge(fact)  # 逐条添加
            
            if assistant_knowledge and assistant_knowledge != "None":
                long_mem.add_assistant_knowledge(assistant_knowledge)
            
            for p in session["details"]:
                p["analyzed"] = True
            session["N_visit"] = 0
            session["L_interaction"] = 0
            session["R_recency"] = 1.0
            session["H_segment"] = 0.0
            session["last_visit_time"] = get_timestamp()
            mid_mem.rebuild_heap()
            mid_mem.save()
            print(f"Update complete: Segment {sid} heat has been reset.")

def generate_system_response_with_meta(query, short_mem, long_mem, retrieval_queue, long_konwledge, client, sample_id, speaker_a, speaker_b, meta_data):
    """
    Generate system response with speaker roles clearly defined.
    Returns response, system_prompt, user_prompt, and context_token count.
    """
    history = short_mem.get_all()
    history_text = "\n".join([
        f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')}\nTime: ({qa.get('timestamp', '')})" 
        for qa in history
    ])
    
    retrieval_text = "\n".join([
        f"【Historical Memory】 {speaker_a}: {page.get('user_input', '')}\n{speaker_b}: {page.get('agent_response', '')}\nTime:({page.get('timestamp', '')})\nConversation chain overview:({page.get('meta_info', '')})\n" 
        for page in retrieval_queue
    ])
    
    profile_obj = long_mem.get_user_profile(sample_id)
    user_profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"
    
    background = f"【User Profile】\n{user_profile_text}\n\n"
    for kn in long_konwledge:
        background += f"{kn['knowledge']}\n"
    background = re.sub(r'(?i)\buser\b', speaker_a, background)
    background= re.sub(r'(?i)\bassistant\b', speaker_b, background)
    assistant_knowledge = long_mem.get_assistant_knowledge()
    assistant_knowledge_text = "【Assistant Knowledge】\n"
    for ak in assistant_knowledge:
        assistant_knowledge_text += f"- {ak['knowledge']} ({ak['timestamp']})\n"
    #meta_data_text = f"【Conversation Meta Data】\n{json.dumps(meta_data, ensure_ascii=False, indent=2)}\n\n"
    assistant_knowledge_text = re.sub(r'\bI\b', speaker_b, assistant_knowledge_text)
    
    # 计算context token - 包含所有用于生成回答的上下文信息
    context_parts = [
        history_text,           # 最近对话历史
        retrieval_text,         # 检索到的历史对话
        background,            # 用户档案和知识库
        assistant_knowledge_text  # 助手知识
    ]
    context_text = "\n".join(context_parts)
    context_token = len(nltk.word_tokenize(context_text)) if context_text else 0
    
    system_prompt = (
        f"You are role-playing as {speaker_b} in a conversation with the user is playing is  {speaker_a}. "
        f"Here are some of your character traits and knowledge:\n{assistant_knowledge_text}\n"
        f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
    )
    
    user_prompt = (
        f"<CONTEXT>\n"
        f"Recent conversation between {speaker_a} and {speaker_b}:\n"
        f"{history_text}\n\n"
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\n"
        f"Characteristics of {speaker_a}:\n"
        f"{background}\n\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat_completion(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=2000)
    return response, system_prompt, user_prompt, context_token

def process_conversation_from_haystack(haystack_sessions, haystack_dates):
    """
    Process conversation data from LME haystack format into memory system format.
    """
    processed = []
    
    # Process each session
    for session_idx, session in enumerate(haystack_sessions):
        timestamp = haystack_dates[session_idx] if session_idx < len(haystack_dates) else ""
        
        for dialog in session:
            role = dialog["role"]
            content = dialog["content"]
            
            # Alternate between user and assistant
            if role == "user":
                processed.append({
                    "user_input": content,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:  # assistant
                if processed:
                    processed[-1]["agent_response"] = content
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": content,
                        "timestamp": timestamp
                    })
    
    return processed

def load_existing_results(output_file):
    """加载已存在的结果文件"""
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"成功加载已有结果，共 {len(results)} 个QA对")
        return results
    except FileNotFoundError:
        print("未找到已有结果文件，将从头开始处理")
        return []
    except Exception as e:
        print(f"加载已有结果时出错：{e}，将从头开始处理")
        return []

def get_processed_question_ids(results):
    """获取已处理的question_id列表"""
    processed_question_ids = set()
    for item in results:
        processed_question_ids.add(item['question_id'])
    return processed_question_ids

def process_single_question(question_data):
    """在单独的进程中处理一个问题，返回该问题的结果。"""
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 为子进程内创建各自的 OpenAI 客户端
        client_local = OpenAIClient(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

        question_id = question_data["question_id"]
        question = question_data["question"]
        original_answer = question_data["answer"]
        question_type = question_data["question_type"]
        haystack_sessions = question_data["haystack_sessions"]
        haystack_dates = question_data["haystack_dates"]

        # 预处理对话
        processing_start = time.time()
        processed_dialogs = process_conversation_from_haystack(haystack_sessions, haystack_dates)
        if not processed_dialogs:
            print(f"问题 {question_id} 没有有效的对话数据，跳过")
            return None

        # 初始化记忆模块（问题级别文件路径，天然无冲突）
        short_mem = ShortTermMemory(max_capacity=1, file_path=f"mem_tmp_lme_final/{question_id}_short_term.json")
        mid_mem = MidTermMemory(max_capacity=2000, file_path=f"mem_tmp_lme_final/{question_id}_mid_term.json")
        long_mem = LongTermMemory(file_path=f"mem_tmp_lme_final/{question_id}_long_term.json")
        dynamic_updater = DynamicUpdate(short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client_local)
        retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10)

        # 存储对话历史
        for dialog in processed_dialogs:
            short_mem.add_qa_pair(dialog)
            if short_mem.is_full():
                dynamic_updater.bulk_evict_and_update_mid_term()
            update_user_profile_from_top_segment(mid_mem, long_mem, question_id, client_local)
        processing_time = time.time() - processing_start

        # 处理问答
        print(f"  [子进程] 处理问题 {question_id}")
        
        retrieval_start = time.time()
        retrieval_result = retrieval_system.retrieve(
            question,
            segment_threshold=0.1,
            page_threshold=0.1,
            knowledge_threshold=0.1,
            client=client_local
        )

        meta_data = {
            "question_id": question_id,
            "question_type": question_type,
            "question_date": question_data.get("question_date", "")
        }
        retrieval_time = time.time() - retrieval_start

        response_start = time.time()
        system_answer, system_prompt, user_prompt, context_token = generate_system_response_with_meta(
            question,
            short_mem,
            long_mem,
            retrieval_result["retrieval_queue"],
            retrieval_result["long_term_knowledge"],
            client_local,
            question_id,
            "user",  # 默认speaker_a
            "assistant",  # 默认speaker_b
            meta_data
        )
        response_time = time.time() - response_start
        
        total_processing_time = response_time + processing_time + retrieval_time

        result = {
            "question_id": question_id,
            "question": question,
            "system_answer": system_answer,
            "original_answer": original_answer,
            "category": question_type,
            "question_date": question_data.get("question_date", ""),
            "timestamp": get_timestamp(),
            "total_processing_time": total_processing_time,
            "processing_time": processing_time,
            "retrieval_time": retrieval_time,
            "response_time": response_time,
            "context_token": context_token,
        }

        print(f"问题 {question_id} 在子进程中处理完成")
        return result
    except Exception as e:
        question_id = question_data.get("question_id", "unknown_question")
        print(f"子进程处理问题 {question_id} 时出错：{e}")
        print(f"错误信息：{e}")
        print(f"错误详情：{str(e)}")
        import traceback
        print("完整错误堆栈：")
        traceback.print_exc()
        return None

def main():
    
    print("开始处理lme数据集...")
    
    # 创建记忆文件存储目录
    os.makedirs("mem_tmp_lme_final", exist_ok=True)
    output_file = f"all_lme_results.json"
    
    # Load lme dataset
    dataset_path = "longmemeval_oracle.json"
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个问题")
    except FileNotFoundError:
        print(f"错误：找不到 {dataset_path} 文件，请确保文件路径正确")
        return
    except Exception as e:
        print(f"加载数据集时发生未知错误：{type(e).__name__}")
        print(f"错误信息：{e}")
        print(f"错误详情：{str(e)}")
        import traceback
        print("完整错误堆栈：")
        traceback.print_exc()
        return
    
    # 加载已有结果（使用分片特定的结果文件）
    results = load_existing_results("all_lme_results.json")
    processed_question_ids = get_processed_question_ids(results)
    
    # 过滤已处理的问题
    remaining_questions = [q for q in dataset if q['question_id'] not in processed_question_ids]
    print(f"已处理问题{len(processed_question_ids)}个: ...{sorted(processed_question_ids)[-3:]}")
    print(f"剩余待处理问题{len(remaining_questions)}个: {[q['question_id'] for q in remaining_questions][-3:]}...")
    
    if not remaining_questions:
        print("没有问题需要处理！")
        return
    
    total_questions = len(remaining_questions)

    # 读取并发度
    try:
        workers = int(os.getenv("NUM_WORKERS", "0"))
    except Exception:
        workers = 0
    if workers <= 0:
        workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"并行处理启动：{total_questions} 个问题，使用 {workers} 个进程")

    # 使用进程池并行处理问题，仅在主进程聚合与落盘
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_question_id = {}
        for idx, question_data in enumerate(remaining_questions):
            question_id = question_data.get("question_id", "unknown_question")
            future = executor.submit(process_single_question, question_data)
            future_to_question_id[future] = question_id

        for future in as_completed(future_to_question_id):
            question_id = future_to_question_id[future]
            try:
                question_result = future.result()
            except Exception as e:
                print(f"收集问题 {question_id} 结果失败：{e}")
                continue

            if not question_result:
                print(f"问题 {question_id} 无结果，跳过保存")
                continue

            results.append(question_result)

            # 增量保存
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"问题 {question_id} 结果已合并并保存到 {output_file}；当前累计 {len(results)} 个问题")
            except Exception as e:
                print(f"保存结果时出错：{e}")
    
    # 最终保存
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"所有问题处理完成！最终结果已保存到 {output_file}")
        print(f"总共处理了 {len(results)} 个问题")
    except Exception as e:
        print(f"最终保存结果时出错：{e}")
        print(f"错误详情：{str(e)}")

if __name__ == "__main__":
    main()
