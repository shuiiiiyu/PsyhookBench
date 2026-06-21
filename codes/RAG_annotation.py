import os, json, base64, io, httpx, pandas as pd, numpy as np, faiss, torch
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel
from modelscope import snapshot_download
from qcloud_cos import CosConfig, CosS3Client

# ===================== 1. 核心路径与配置 =====================
GOLDEN_REASON_CSV = r"E:\Phyhookbench\TEST\postid.csv"
GOLDEN_TITLE_LIST = [r"E:\Phyhookbench\TEST\DATA_SPLIT\train.csv", r"E:\Phyhookbench\TEST\DATA_SPLIT\test.csv"]
TASK_CSV = r"E:\Phyhookbench\TEST\split_parts\2500_part_5.csv"

WORKSPACE_DIR = Path(r"E:\phyhookbench\TEST")
RESULT_DIR = WORKSPACE_DIR / "results_300_5"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

API_KEY = "sk-VJICNERX531a3a4b2b14T3BLbKFJD0458ACf5c4d4bc29872"
BASE_URL = "https://api.ohmygpt.com/v1"
MODEL_NAME = "gpt-4o-2024-11-20"

COS_CONFIG = {
    'SecretId': xxxx,
    'SecretKey': xxxx,
    'Region': xxx,
    'Bucket': xxx
}

# ===================== 2. 提示词 (保持原样) =====================
BASE_HOOK_DEFINITIONS = r"""

You are a social media content analyst. Analyze the following post (Title and Cover Image) for psychological hooks. 
For each hook, output 1 if present, 0 if not.

[Definitions]
hook1: Fear Of Missing Out (FOMO)
核心定义：内容示意了“如果你不做或错过某事，就会有什么样的损失”，以激发受众的担忧和增长紧张情绪。
操作化判断：内容中是否包含/传达了：“不行动”的线索 AND “不行动”的“代价”、“后果”
参照线索：
- 不行动的线索：不看/不听/不做/不像帖子中这样的话...
- “不行动”的“代价”、“后果”包括：破产/分手/失败...等负面意向
Instruction: Do NOT label '1' just for negative words (sad, bad). In addition to negative words, there also needs to be clues about missing out or inaction.


hook2: Gain Appeal
核心定义：内容强调了“通过此内容信息能获得什么好处”，以激发受众本能的获取动机。
操作化判断：内容中是否包含/传达了：内容能带来的好处
参照线索：
- 好处：金钱（省钱、赚钱）、时间（节省时间、提高效率）、健康（变瘦、变美）、技能（速成、精通）、情感（快乐、安心）…等正面意向


hook3: Information-gap
核心定义：内容在标题、封面本可以完整表达、概括其内容信息的情况下，却故意挖空部分信息，以引导观众点开去找。
操作化判断：作者是否有意图：故意隐藏信息
参照线索：
- 自问自答（属于通过问题探针形式的特殊信息缺口类型）：在标题或封面提出问题，在点击后的内容中回答
- 遮挡关键信息：用马赛克、贴纸等遮挡标题或封面的关键部分
- 话只说一半：用省略号、中断、留白等方式截断句子或故事
- 设置悬念：使用各种形式对缺失的信息进行铺垫、渲染
- 只抛出情境：例如“当...”、“pov：…”
- 指代不明：用"这个""那个"指代，但不知道指代的到底是什么
（或：以上没有列举但符合核心定义的线索）
注意：Almost all social media cards have titles. Do NOT label 'Information Gap' just because there is a title.
Core defined boundaries (to avoid cognitive divergence)：
1.Exclusionary boundaries: Incomplete information caused by the limitations of preview section in terms of length and display format (such as the upper limit of title characters, cover image size) does not belong to information gaps;
2.Initiative boundaries: The gap is subjectively and deliberately designed by the creator, rather than being objectively restricted in content expression. The core is "could have finished but intentionally didn't";
3.Core boundaries: What is missing is the core information of the content (such as results, answers, key details, core conclusions), not insignificant auxiliary information.
排除线索：单纯求助帖子、封面已经对标题的内容做了回答的


hook4: Anomaly and novelty
核心定义：内容被故意包装成惊人、违反常理或罕见、新奇的，以激发受众的好奇心。
操作化判断：内容中是否包含了：表现惊人反常的短语 OR 表现罕见新奇的短语
参照线索：
- 表现惊人反常的短语：竟然/居然/没想到/不可思议/罕见/第一次见/震惊/惊呆/看傻/刷新三观/刷新认知/不可思议/神奇…
- 表现罕见新奇的短语包括：
  - 直接声称新奇的词：独创/别具一格/标新立异/新颖…
  - 极限词：最/顶/超/第一/史诗级…
  - 稀缺性：唯一/只有/限定/鲜见/偶发/孤例孤品/小众/冷门/千年一遇…
（注意：如果内容本身够反常新奇但没有被作者包装，则不属于此类型；此外，常规分享、情绪宣泄中只有包含了上述词汇或类似词汇的才属于此类型）
Instruction: Do NOT label '1' just for What you consider novel, interesting, or contrary to common sense. What we are looking for is the action and intention of the author in packaging the content into a striking contrast or rare novelty.


hook5: Perceptual Contrast
核心定义：内容通过视觉或者文字将两种或多种形成反差的状态或事物放在一起，以激发受众的探索欲。
操作化判断：看图片或文字中有没有两个及多个有明显反差的对比项,但是一定要反差，有比较但差别不大的不算
参照线索：
- 对比项包括：前后/左右/正反/好坏/预期与现实/别人与自己…等
- 对比形式包括：文本和文本的语义反差、图像之间的视觉反差、文本与图像之间的反差
注意：需要识别感知到任何可能构成反差的线索


hook6: Ingroup Identification / Outgroup Distinction
核心定义：内容通过群体标签，激发某一群体内的认同、归属；或激发对某一群体的排斥、调侃。
操作化判断：内容中是否：出现群体标签 AND（出现归属/排斥态度 OR 行动召唤 OR 群体共性）
参照线索：
- 群体标签：提及某一群体，如
  - 提到：和种族/国家/民族/宗教/地域…等相关的名词
  - 提到：和学校/公司/组织/机构/社区…等相关的名词
  - 提到：和年龄/性别/职业…等相关的名词
  - 以及其他：某种爱好标签/某种性格标签/某种星座标签/某种mbti标签…等
  （只要出现的短语可以在人群中划分出一个群体和另外的人，即可）
- 归属/排斥态度：表现出骄傲、自豪、认同、共情…；或：鄙视、调侃、讽刺…等
- 行动召唤：“...必看”、“是...就点赞”、“...们行动起来”…等
- 群体共性：“每个...都经历过”、“...的日常”、“...都懂”…以及体现在画面中的共性特征等
Instruction: Be SENSITIVE. If you suspect the content contains any jargon or visual style specific to a niche group, even if you are not 100% sure, Please also take it into consideration.


hook7: Social Comparison
核心定义： 内容通过直接使用明显的比较词、展示差距、或展示社会比较后的某种态度等，来引发受众参与比较。
操作化判断：
内容中是否：明显的比较行为词 OR（展示差距 + 展示了社会比较后态度）
参照线索：
- 明显的比较行为词：比.../更/VS/不如.../碾压/秒杀…等
- 差距：能力差距（技能、成就、任务绩效等）/个人特质差距（外貌、性格、天赋、身高等）/资源差距（财富、生活水平、地位高低、权利差别等）…
- 比较后的态度： -向上比较：嫉妒、自卑等消极态度；认可、激励上进等积极态度
                                  -向下比较：炫耀、优越感等消极态度；珍惜、知足等积极态度
注意：
1.差距的参照点可以是内容中直接展现的对象，也可以是大众默认的普遍水平；因此请排除常规的普遍水平的分享类笔记，这不构成社会比较。
2.社会比较的比较对象是自己与别人比较、别人与别人比较、群体与群体比较（不包含自己与自己比较、单纯的物品比较）


hook8: Authority Endorsement
核心定义：文字或图片通过各种有说服力的信源背书来豁免受众的质疑成本，从而引导受众信服或模仿。
操作化判断：文字或图片中是否出现：信源背书
参照线索：
- 权威信源包括：专家/教授/机构/名人/研究/排名/认证/奖项/数字…等
  - 如：哈佛大学研究/据...报道/...专家说/FDA认证/某某明星同款/青岛第一的/22w人看过的…
注意：凡是能增加可信度的线索都在范围内，先标注起来比漏标要好。

[Note] 
1. We are in the discovery phase. If a post shows even a slight tendency or subtle hint of a hook, please lean towards labeling it as 1.
2. First, find reasons to label it as 1; only if no clues exist, label it as 0.
3. INDEPENDENT JUDGMENT: The [Reference Examples] provided below are for reference and reasoning logic understanding ONLY. Judge the target post independently based on its own content.
4. Definitions是最重要的，记得以看定义为主，参考样本仅辅助作用

[Thought Process required]
Before making a decision, you MUST:
1. Identify visual elements in the image (stickers, mosaics, layout).
2. Analyze the semantic intent of the title.
3. Combine them to consider the author's creative intent.
4. Compare them with the core definitions.


[Json Output Format]
Return the results in JSON format. Ensure 'reasoning' comes FIRST:
{
  "reasoning": "Step-by-step Chinese analysis of why hook1-8 is/isn't present (<50 chars)",
  "h1":0/1, "h2":0/1, "h3":0/1, "h4":0/1, "h5":0/1, "h6":0/1, "h7":0/1, "h8":0/1
}

"""

# ===================== 3. 工具函数 =====================
def get_cos_client():
    config = CosConfig(Region=COS_CONFIG['Region'], SecretId=COS_CONFIG['SecretId'], SecretKey=COS_CONFIG['SecretKey'])
    return CosS3Client(config)

def parse_expert_cell(cell_value):
    val = str(cell_value).strip()
    reason = val.split("：")[-1] if "：" in val else (val.split(":")[-1] if ":" in val else "")
    if val == "高共识" or val == "1" or "边缘案例1" in val:
        return 1, (reason if reason != val else "专家判定符合定义")
    return 0, (reason if reason != val else "不符合特征")

# ===================== 4. 主流程 =====================
def run_pipeline():
    # A. 知识库准备 (BGE-M3)
    print("📂 步骤 1: 构建 BGE-M3 专家知识库...")
    df_titles = pd.concat([pd.read_csv(f, dtype={'post_id': str}) for f in GOLDEN_TITLE_LIST])
    df_titles['post_id'] = df_titles['post_id'].astype(str).str.strip().str.lower()
    df_titles = df_titles.drop_duplicates('post_id')
    df_reason = pd.read_csv(GOLDEN_REASON_CSV, dtype={'post_id': str})
    df_reason['post_id'] = df_reason['post_id'].astype(str).str.strip().str.lower()
    df_kb = pd.merge(df_reason, df_titles[['post_id', 'title']], on='post_id', how='inner').reset_index(drop=True)

    for i in range(1, 9):
        res = df_kb[str(i)].apply(parse_expert_cell)
        df_kb[f'h{i}_label'], df_kb[f'h{i}_reason'] = res.apply(lambda x: x[0]), res.apply(lambda x: x[1])

    model_path = snapshot_download('BAAI/bge-m3')
    model = BGEM3FlagModel(model_path, use_fp16=True)
    kb_vectors = model.encode(df_kb['title'].fillna('').tolist(), batch_size=16)['dense_vecs']
    index = faiss.IndexFlatIP(kb_vectors.shape[1])
    index.add(kb_vectors.astype("float32"))

    # B. 5 轮标注执行
    df_task = pd.read_csv(TASK_CSV, dtype={'post_id': str})
    df_task['post_id'] = df_task['post_id'].astype(str).str.strip().str.lower()
    task_ids = df_task['post_id'].tolist()
    
    ai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=httpx.Client(timeout=60.0))
    cos_client = get_cos_client()
    
    # 存储所有人的投票和理由
    # 结构: {pid: { "h1_votes": 0, "round1_reason": "", ... }}
    audit_data = {pid: {"post_id": pid} for pid in task_ids}
    
    for r in range(1, 6):
        print(f"🚀 正在运行第 {r}/5 轮标注 (Temperature 0.5)...")
        for pid in tqdm(task_ids, desc=f"Round {r}"):
            try:
                task_row = df_task[df_task['post_id'] == pid].iloc[0]
                title = str(task_row.get('title', ''))
                
                # RAG
                q_vec = model.encode([title])['dense_vecs']
                _, I = index.search(q_vec.astype('float32'), 4)
                example_text = "\n[Reference Examples]\n"
                for idx in I[0]:
                    ref = df_kb.iloc[idx]
                    labels = ", ".join([f"h{i}:{ref[f'h{i}_label']}" for i in range(1, 9)])
                    example_text += f"- Title: {ref['title']}\n- Labels: {labels}\n---\n"

                # Cloud Image
                cos_obj = cos_client.get_object(Bucket=COS_CONFIG['Bucket'], Key=f"downloads/{pid}/cover.jpg")
                img_b64 = base64.b64encode(cos_obj['Body'].get_raw_stream().read()).decode("utf-8")

                response = ai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": BASE_HOOK_DEFINITIONS + example_text},
                        {"role": "user", "content": [{"type": "text", "text": f"待标注标题: {title}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
                    ],
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
                
                res = json.loads(response.choices[0].message.content)
                
                # 记录理由
                audit_data[pid][f"R{r}_Reason"] = res.get("reasoning", "")
                # 记录票数
                for i in range(1, 9):
                    vote_key = f"h{i}_votes"
                    if vote_key not in audit_data[pid]: audit_data[pid][vote_key] = 0
                    
                    k = f"h{i}" if f"h{i}" in res else str(i)
                    if res.get(k) == 1:
                        audit_data[pid][vote_key] += 1
                        
            except Exception as e:
                audit_data[pid][f"R{r}_Reason"] = f"ERROR: {e}"

    # C. 生成详细汇总表
    df_audit = pd.DataFrame(list(audit_data.values()))
    # 合并原始任务信息（标题等）
    df_audit = pd.merge(df_task, df_audit, on='post_id', how='left')
    df_audit.to_csv(RESULT_DIR / "full_consensus_audit_raw.csv", index=False, encoding='utf-8-sig')

    # D. 核心筛选逻辑：根据你的票数规则分发 8 个 Hook 的复核表
    print("📊 正在生成差异化复核清单...")
    
    # 规则配置
    RULES = {
        'h1_votes': [3, 4, 5], 'h4_votes': [3, 4, 5],
        'h2_votes': [2, 3], 'h3_votes': [2, 3], 'h5_votes': [2, 3], 'h6_votes': [2, 3],
        'h7_votes': [3], 'h8_votes': [3]
    }

    for hook_vote_col, target_votes in RULES.items():
        # 筛选符合票数规则的行
        df_hook_review = df_audit[df_audit[hook_vote_col].isin(target_votes)].copy()
        
        if not df_hook_review.empty:
            hook_name = hook_vote_col.replace("_votes", "")
            file_name = f"REVIEW_NEEDED_{hook_name}.csv"
            
            # 整理列顺序：ID -> 标题 -> 当前Hook票数 -> 5轮理由
            cols_to_keep = ['post_id', 'title', hook_vote_col, 'R1_Reason', 'R2_Reason', 'R3_Reason', 'R4_Reason', 'R5_Reason']
            df_hook_review[cols_to_keep].to_csv(RESULT_DIR / file_name, index=False, encoding='utf-8-sig')
            print(f"  - 已生成 {hook_name} 复核表: {len(df_hook_review)} 条")

    print("-" * 30)
    print(f"✅ 所有流程完成！请在 {RESULT_DIR} 查看 8 张 Hook 专用复核表。")

if __name__ == "__main__":
    run_pipeline()
