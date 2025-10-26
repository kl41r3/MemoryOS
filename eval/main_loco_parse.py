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
    return response, system_prompt, user_prompt

def process_conversation(conversation_data):
    """
    Process conversation data from locomo10 format into memory system format.
    Handles both text-only and image-containing messages.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]
    
    # Find all session keys
    session_keys = [key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")]
    
    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        
        for dialog in conversation_data[session_key]:
            speaker = dialog["speaker"]
            text = dialog["text"]
            
            # Handle image content if present
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"
            
            # Alternate between speakers as user and assistant
            if speaker == speaker_a:
                processed.append({
                    "user_input": text,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": text,
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

def get_processed_samples(results):
    """获取已处理的样本ID列表"""
    processed_samples = set()
    for item in results:
        processed_samples.add(item['sample_id'])
    return processed_samples

def process_single_sample(sample):
    """在单独的进程中处理一个样本，返回该样本的结果列表。"""
    try:
        # 为子进程内创建各自的 OpenAI 客户端
        client_local = OpenAIClient(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

        conversation_data = sample["conversation"]
        qa_pairs = sample["qa"]
        sample_id = sample.get("sample_id", "unknown_sample")

        # 预处理对话
        processed_dialogs = process_conversation(conversation_data)
        if not processed_dialogs:
            print(f"样本 {sample_id} 没有有效的对话数据，跳过")
            return []

        speaker_a = conversation_data["speaker_a"]
        speaker_b = conversation_data["speaker_b"]

        # 初始化记忆模块（样本级别文件路径，天然无冲突）
        short_mem = ShortTermMemory(max_capacity=1, file_path=f"mem_tmp_loco_final/{sample_id}_short_term.json")
        mid_mem = MidTermMemory(max_capacity=2000, file_path=f"mem_tmp_loco_final/{sample_id}_mid_term.json")
        long_mem = LongTermMemory(file_path=f"mem_tmp_loco_final/{sample_id}_long_term.json")
        dynamic_updater = DynamicUpdate(short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client_local)
        retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10)

        # 存储对话历史
        for dialog in processed_dialogs:
            short_mem.add_qa_pair(dialog)
            if short_mem.is_full():
                dynamic_updater.bulk_evict_and_update_mid_term()
            update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client_local)

        # 处理问答
        sample_results = []
        qa_count = len(qa_pairs)
        for qa_idx, qa in enumerate(qa_pairs):
            print(f"  [子进程] {sample_id} 问答 {qa_idx + 1}/{qa_count}")
            question = qa["question"]
            original_answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa.get("evidence", "")
            if original_answer == "":
                original_answer = qa.get("adversarial_answer", "")

            retrieval_result = retrieval_system.retrieve(
                question,
                segment_threshold=0.1,
                page_threshold=0.1,
                knowledge_threshold=0.1,
                client=client_local
            )

            meta_data = {
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "category": category,
                "evidence": evidence
            }

            system_answer, system_prompt, user_prompt = generate_system_response_with_meta(
                question,
                short_mem,
                long_mem,
                retrieval_result["retrieval_queue"],
                retrieval_result["long_term_knowledge"],
                client_local,
                sample_id,
                speaker_a,
                speaker_b,
                meta_data
            )

            sample_results.append({
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "system_answer": system_answer,
                "original_answer": original_answer,
                "category": category,
                "evidence": evidence,
                "timestamp": get_timestamp(),
            })

        print(f"样本 {sample_id} 在子进程中处理完成，共生成 {len(sample_results)} 个QA对")
        return sample_results
    except Exception as e:
        sample_id = sample.get("sample_id", "unknown_sample")
        print(f"子进程处理样本 {sample_id} 时出错：{e}")
        print(f"错误信息：{e}")
        print(f"错误详情：{str(e)}")
        import traceback
        print("完整错误堆栈：")
        traceback.print_exc()
        return

def main():

    print("开始处理整个locomo10数据集...")
    
    # 创建记忆文件存储目录
    os.makedirs("mem_tmp_loco_final", exist_ok=True)
    output_file = "all_loco_results.json"
    
    # 加载已有结果
    results = load_existing_results(output_file)
    processed_samples = get_processed_samples(results)
    
    # Load locomo10 dataset
    try:
        with open("locomo10.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 个样本")
    except FileNotFoundError:
        print("错误：找不到 locomo10.json 文件，请确保文件在当前目录中")
        return
    except Exception as e:
        print(f"加载数据集时发生未知错误：{type(e).__name__}")
        print(f"错误信息：{e}")
        print(f"错误详情：{str(e)}")
        import traceback
        print("完整错误堆栈：")
        traceback.print_exc()
        return
    
    # 过滤样本
    remaining_samples = [sample for sample in dataset if sample['sample_id'] not in processed_samples]
    print(f"已处理样本: {sorted(processed_samples)}")
    print(f"剩余待处理样本: {[s['sample_id'] for s in remaining_samples]}")
    
    print(f"待处理样本数量: {len(remaining_samples)}")
    
    if not remaining_samples:
        print("没有样本需要处理！")
        return
    
    total_samples = len(remaining_samples)

    # 读取并发度
    try:
        workers = int(os.getenv("NUM_WORKERS", "0"))
    except Exception:
        workers = 0
    if workers <= 0:
        workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"并行处理启动：{total_samples} 个样本，使用 {workers} 个进程")

    # 使用进程池并行处理样本，仅在主进程聚合与落盘
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_sample_id = {}
        for idx, sample in enumerate(remaining_samples):
            sample_id = sample.get("sample_id", "unknown_sample")
            print(f"提交样本 {idx + 1}/{total_samples}: {sample_id}")
            future = executor.submit(process_single_sample, sample)
            future_to_sample_id[future] = sample_id

        for future in as_completed(future_to_sample_id):
            sample_id = future_to_sample_id[future]
            try:
                sample_results = future.result()
            except Exception as e:
                print(f"收集样本 {sample_id} 结果失败：{e}")
                continue

            if not sample_results:
                print(f"样本 {sample_id} 无新增结果，跳过保存")
                continue

            results.extend(sample_results)

            # 增量保存
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"样本 {sample_id} 结果已合并并保存到 {output_file}；当前累计 {len(results)} 个QA对")
            except Exception as e:
                print(f"保存结果时出错：{e}")
    
    # 最终保存
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"所有样本处理完成！最终结果已保存到 {output_file}")
        print(f"总共处理了 {len(results)} 个QA对")
    except Exception as e:
        print(f"最终保存结果时出错：{e}")
        print(f"错误详情：{str(e)}")

if __name__ == "__main__":
    main()
