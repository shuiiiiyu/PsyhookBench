import os, json, base64, io, httpx, pickle, time, shutil, re
import pandas as pd
import numpy as np
import faiss
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from qcloud_cos import CosConfig, CosS3Client
import cn_clip.clip as clip
from cn_clip.clip.utils import load_from_name
from modelscope.hub.snapshot_download import snapshot_download
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report

# ===================== 1. 核心路径配置 =====================
GOLDEN_LABEL_CSV = "postid.csv" 
GOLDEN_TITLE_CSV = "541_raw_data.csv"
TASK_CSV = "2500_raw_data.csv"
EXPERT_CSV = "expert_543.csv"
WORKSPACE_DIR = Path("WORKSPACE")
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = WORKSPACE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

API_KEY = "sk-xxxx"
BASE_URL = "https://api.ohmygpt.com/v1"
MODEL_NAME = "gpt-4o-2024-11-20"

COS_CONFIG = {
    'SecretId': 'xxx',
    'SecretKey': 'xxx',
    'Region': 'xxx',
    'Bucket': 'xxx'
}

IMAGE_CACHE = {}

# ===================== 2. 提示词模块 (一个字未改) =====================
BASE_HOOK_DEFINITIONS = """
You are a social media content analyst. Analyze the following post (Title and Cover Image) for psychological hooks. 
For each hook, output 1 if present, 0 if not.

[Definitions]
hook1: Fear Of Missing Out (FOMO)
- Focus: "If you don't act, you lose."
- Typical Phrases: ，“再不看就删了”, “你还没用XX吗”, “错过等一年”, “不懂的人都在亏”, “不学你就亏大了”, “后悔没早看”.
Core Definition: The content expresses "if you don't do something, there will be certain consequences (losses / regrets)" to arouse the audience's concerns and increase tension.
Operational Judgment:
Does the content contain/convey: clues of "inaction" AND "costs" or "consequences" of "inaction"
Reference Clues:
Clues of "inaction": words like not watching / not listening / not doing / not acting like what is mentioned in the post...
"Costs" and "consequences" of "inaction" include: negative intentions such as bankruptcy / breakup / failure...


hook2: Gain Appeal
- Focus: "Do this to get benefits (money, time, ease)."
- Typical Phrases: “分享”，“攻略”，“如何”，“测评”，“速通”，“每天多赚XXX元”, “5分钟学会XX”, “教你白嫖XX”, “看完这篇省下XXX”, “轻松搞定XX”.
Core Definition: The content emphasizes "what benefits can be obtained through this content information" to stimulate the audience's inherent motivation to acquire.
Operational Judgment:
Does the content contain/convey: the benefits that the content can bring
Reference Clues:
Benefits: Money (saving money, making money), time (saving time, improving efficiency), health (losing weight, becoming more beautiful), skills (quick learning, mastery), emotions (happiness, peace of mind)... and other positive intentions


hook3: Information-gap
- Focus: Intentional omission to create curiosity.
- Mandatory Clues: If the title or cover image contains "......" (ellipsis) or "?" (question mark) or "...", you MUST label this hook as 1.
- Typical Phrases: “这样”，“是谁还没”，“看到最后我惊了”, “结果让我沉默”, “原因你绝对想不到”, “到底怎么回事”, “不是，他怎么...”, “竟然是因为这个”，“竟然是因为那个”.
Core Definition: When the content could have fully conveyed information through the title or cover, it intentionally omits part of the information to guide the audience to click and find it.
Operational Judgment:
Does the content create: intentionally hidden information
Reference Clues:
Self-questioning and self-answering: Raising a question in the title or cover and answering it in the content
Obscuring key information: Using mosaics, stickers, etc., to cover key parts of the title or cover
Leaving words half-said: Truncating sentences or stories with ellipsis, interruptions, leaving blank space, etc.
Setting suspense: Using various forms to render the missing information
Only presenting scenarios: For example, "When...", "pov:..."
Ambiguous reference: Using words like "this" or "that" or “这个” or "这……" or "那……" to refer to something, but it is unclear what exactly they refer to
(Or: clues not listed above but conforming to the core definition)


hook4: Anomaly and novelty
- Focus: Astonishing, counterintuitive, or rare content.
- Typical Phrases: “竟然”，“居然”，“第一个”，“最”，“小众”，“XX居然能YY？”, “医生让我吃XX”, “常识骗了你”, “看似失败其实是”, “违反常理”, “首次公开”, “你没见过的”, “罕见现象”.
Core Definition: Content is deliberately packaged to be astonishing, counterintuitive, rare, or novel in order to arouse the audience's curiosity.
Operational Judgment:
Does the content contain: phrases that express astonishing abnormality OR phrases that express rarity and novelty
Reference Clues:
Phrases expressing astonishing abnormality: actually/unexpectedly (竟然), unexpectedly (居然), didn't expect (没想到), incredible (不可思议), rare (罕见), seeing for the first time (第一次见), shocked (震惊), stunned (惊呆), dumbfounded (看傻), refreshing one's outlook on life (刷新三观), refreshing one's cognition (刷新认知), incredible (不可思议), magical (神奇)…
Phrases expressing rarity and novelty include: -Words directly claiming novelty: original (独创), unique (别具一格), novel and different (标新立异), novel (新颖)…Extreme words: most (最), top (顶), super (超), first (第一), epic-level (史诗级)…
Scarcity: only (唯一), only (只有), limited (限定), rarely seen (鲜见), occasional (偶发), unique case/unique item (孤例孤品), niche (小众), unpopular (冷门), once in a millennium (千年一遇)…
(Note: Regardless of the content itself, it counts as such as long as it is packaged to be abnormal or novel; conversely, if the content itself is abnormal or novel enough but not packaged by the author, it does not count.)


hook5: Perceptual Contrast
- Focus: Visual or semantic contrast (Before/After, VS).
- Typical Phrases: “别人vs我”,“前后对比惊人”, “ vs 差别”, “几千和几万的”, “VS”, “以前现在对比”.
Core Definition: Content places two or more contrasting states or things together visually or textually to stimulate the audience's desire to explore.
Operational Judgment:
Does the content contain/convey: two or more contrasting items with obvious differences
Reference Clues:
Contrasting items include: front and back/left and right/positive and negative/good and bad/expectation and reality/others and oneself... etc.
Forms of contrast include: semantic contrast between texts, visual contrast between images, contrast between text and image
Note: It is necessary to identify and perceive any clues that may constitute a contrast


hook6: Ingroup Identification / Outgroup Distinction
- Focus: Group labels and belonging.
Core Definition: Content uses group labels to arouse a sense of identity and belonging within a certain group; or to arouse rejection and ridicule towards a certain group.
Operational Judgment:
In the content: Does it contain group labels AND (display attitudes of belonging/rejection OR calls to action OR group commonalities)
Reference Clues:
Group Labels: Mention of a certain group, such as
Mention: nouns related to race/country/ethnicity/religion/region...
Mention: nouns related to school/company/organization/institution/
Mentioned: nouns related to age/gender/occupation... etc.
And others: certain hobby tags/certain personality tags/certain zodiac signs/certain MBTI tags... etc.
(As long as the appearing phrase can divide a group from others in the crowd, it is acceptable)
Attitude of belonging/exclusion: Showing pride, honor, identification, empathy...; or: disdain, teasing, satire... etc.
Call to action: "...must watch", "like if you are...", "...let's take action"... etc.
Group commonalities: "Every... has experienced it", "the daily life of...", "...all understand"... and common characteristics reflected in the pictures, etc.
If the title or cover mentions ANY specific group of people, you MUST label this hook as 1. No other attitudes or calls to action are required.

hook7: Social Comparison
- Focus: Gaps between self and others.
- Typical Phrases: “别人家的”, “差距有多大”, “羡慕别人的”, “我VS别人的”, “水平对比”.
Core Definition: Content triggers the audience to engage in comparison by directly using obvious comparative words, showing gaps and displaying certain attitudes after social comparison.
Operational Judgment:
In the content: Are there obvious comparative action words OR (display of gaps AND display of attitudes after social comparison)
Reference Clues:
Obvious comparative action words: than... (比...), more (更), VS, not as good as... (不如...), crush (碾压), instantly defeat... (秒杀…) etc.
Gaps: Ability gaps (skills, achievements, task performance, etc.) / Self-trait gaps (appearance, personality, talent, height, etc.) / Resource gaps (wealth, living standards, status, power differences, etc.)...
Attitudes after comparison:Upward comparison: Negative attitudes such as jealousy and inferiority; Positive attitudes such as recognition and motivation to make progress.
Downward comparison: Negative attitudes such as showing off and a sense of superiority; Positive attitudes such as cherishing and being content.
Note: The reference point for the gap can be explicit (e.g., a direct object is shown) or implicit (e.g., the generally accepted level by the public).


hook8: Authority Endorsement
- Focus: Persuasive sources.
- Typical Phrases: “同款”，“专家”, “央视”, “官方”, “哈佛”, “机构”, .
Core Definition: Content uses various persuasive source endorsements to exempt the audience from the cost of questioning, thereby guiding the audience to be convinced or imitate.
Operational Judgment:
Does the content contain: source endorsement
Reference Clues:
Authoritative sources include: experts/professors/institutions/celebrities/research/rankings/certifications/awards/numbers... etc.For example: Harvard University research/According to... reports/... experts say/FDA certification/XX celebrity's same style/Qingdao's top 1…/220,000 people have viewed…

[Note] 
1. We are in the discovery phase. If a post shows even a slight tendency or subtle hint of a hook, please lean towards labeling it as 1.
Do not be overly restricted by the 'Exclusionary boundaries' unless you are 100% sure it is a normal preview.
2. First, find reasons to label it as 1; only if no clues exist, label it as 0.
3. INDEPENDENT JUDGMENT: The [Reference Examples] provided below are retrieved based on semantic similarity and are for reference and reasoning logic understanding ONLY. There is NO necessary correlation between the labels of reference examples and the target post. You MUST judge the target post independently based on its own content.

[Thought Process required]
Before making a decision, you MUST:
1. Identify visual elements in the image (stickers, mosaics, layout).
2. Analyze the semantic intent of the title.
3. Compare them with the core definitions.

[Json Output Format]
Return the results in JSON format. Ensure 'reasoning' comes FIRST:
{
  "reasoning": "Step-by-step Chinese analysis of why hook3,hook5,hook6,hook7 is/isn't present (<50 chars)",
  "h1":0/1, "h2":0/1, "h3":0/1, "h4":0/1, "h5":0/1, "h6":0/1, "h7":0/1, "h8":0/1
}"""

# ===================== 3. 工具函数 =====================

def clean_id(x): return str(x).strip().lower()

def get_cos_client():
    config = CosConfig(Region=COS_CONFIG['Region'], SecretId=COS_CONFIG['SecretId'], SecretKey=COS_CONFIG['SecretKey'])
    return CosS3Client(config)

def load_clip_model():
    print("🚀 正在加载 Chinese-CLIP 多模态编码模型...")
    model_dir = snapshot_download("AI-ModelScope/chinese-clip-vit-base-patch16")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device)
    return model.eval(), preprocess, device

def get_image_b64_from_cos(cos_client, path):
    if path in IMAGE_CACHE: return IMAGE_CACHE[path]
    try:
        cos_obj = cos_client.get_object(Bucket=COS_CONFIG['Bucket'], Key=path)
        img_data = cos_obj['Body'].get_raw_stream().read()
        b64 = base64.b64encode(img_data).decode("utf-8")
        IMAGE_CACHE[path] = b64
        return b64
    except: return None

def generate_validation_report(final_df, expert_csv_path):
    try:
        df_gt = pd.read_csv(expert_csv_path)
        hook_cols = [f'h{i}' for i in range(1, 9)]
        df_gt.columns = df_gt.columns.str.strip().lower()
        for col in hook_cols:
            if col in df_gt.columns:
                df_gt[col] = df_gt[col].apply(lambda x: 1 if '1' in str(x) or '高共识' in str(x) else 0)
        merged = pd.merge(df_gt[['post_id'] + hook_cols], final_df, on='post_id', suffixes=('_gt', '_pred'))
        y_true = merged[[f'{h}_gt' for h in hook_cols]].values
        y_pred = merged[[f'{h}_pred' for h in hook_cols]].values
        print("\n" + "📊" * 15 + " 最终效度验证报告 " + "📊" * 15)
        print(f"✅ 对齐样本总数: {len(merged)}")
        print(f"🎯 完全匹配率 (EMR): {accuracy_score(y_true, y_pred):.4f}")
        print(f"📉 汉明损失 (Hamming Loss): {hamming_loss(y_true, y_pred):.4f}")
        print(f"⚖️ 宏平均 F1 (Macro-F1): {f1_score(y_true, y_pred, average='macro'):.4f}")
        print("-" * 60)
        for i, h in enumerate(hook_cols, 1):
            f1 = f1_score(y_true[:, i-1], y_pred[:, i-1])
            print(f"[Hook {i}] F1 Score: {f1:.4f}")
    except Exception as e: print(f"⚠️ 无法生成比对报告: {e}")

# ===================== 4. 主流程 =====================

def run_pipeline():
    total_calls = 0
    invalid_outputs = 0
    
    print("📂 步骤 1: 正在构建统一元数据池...")
    df_golden = pd.read_csv(GOLDEN_TITLE_CSV, dtype={'post_id': str})
    df_task = pd.read_csv(TASK_CSV, dtype={'post_id': str})
    df_label_base = pd.read_csv(GOLDEN_LABEL_CSV, dtype={'post_id': str})
    
    unified_meta = {}
    for df in [df_golden, df_task]:
        for _, row in df.iterrows():
            pid = clean_id(row['post_id'])
            unified_meta[pid] = {
                "title": str(row.get('title', '')),
                "cover_path": f"downloads/{pid}/cover.jpg",
                "label_info": ""
            }
    
    # 注入金标准标签信息用于 RAG 文本展示
    for _, row in df_label_base.iterrows():
        pid = clean_id(row['post_id'])
        if pid in unified_meta:
            labels = [f"h{i}:{row[str(i)]}" for i in range(1, 9) if str(i) in df_label_base.columns]
            unified_meta[pid]["label_info"] = ", ".join(labels)

    print("🧠 步骤 2: 正在提取双权重特征索引...")
    model, preprocess, device = load_clip_model()
    cos_client = get_cos_client()
    golden_ids = df_golden['post_id'].map(clean_id).tolist()
    
    vecs_82, vecs_55 = [], []
    valid_ids_in_index = []
    
    for pid in tqdm(golden_ids, desc="Encoding Golden Base"):
        meta = unified_meta[pid]
        try:
            cos_obj = cos_client.get_object(Bucket=COS_CONFIG['Bucket'], Key=meta['cover_path'])
            img_raw = cos_obj['Body'].get_raw_stream().read()
            img_input = preprocess(Image.open(io.BytesIO(img_raw))).unsqueeze(0).to(device)
            text_input = clip.tokenize([meta['title']]).to(device)
            
            with torch.no_grad():
                i_feat = model.encode_image(img_input)
                i_feat /= i_feat.norm(dim=-1, keepdim=True)
                t_feat = model.encode_text(text_input)
                t_feat /= t_feat.norm(dim=-1, keepdim=True)
                
                v82 = (0.8 * t_feat + 0.2 * i_feat)
                v82 /= v82.norm(dim=-1, keepdim=True)
                v55 = (0.5 * t_feat + 0.5 * i_feat)
                v55 /= v55.norm(dim=-1, keepdim=True)
                
                vecs_82.append(v82.cpu().numpy().flatten())
                vecs_55.append(v55.cpu().numpy().flatten())
                valid_ids_in_index.append(pid)
        except: continue

    idx_82 = faiss.IndexFlatIP(len(vecs_82[0]))
    idx_82.add(np.asarray(vecs_82, dtype="float32"))
    idx_55 = faiss.IndexFlatIP(len(vecs_55[0]))
    idx_55.add(np.asarray(vecs_55, dtype="float32"))

    print("🤖 步骤 3: 分组启动多模态 RAG 标注...")
    ai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=httpx.Client(timeout=60.0))
    task_ids = df_task['post_id'].map(clean_id).tolist()
    final_decisions = {pid: {"post_id": pid} for pid in task_ids}
    
    # 机制分组逻辑
    groups_config = [
        {"hooks": [1, 2, 3, 4, 6], "index": idx_82, "weight": (0.8, 0.2), "name": "Semantic-Heavy"},
        {"hooks": [5, 7, 8], "index": idx_55, "weight": (0.5, 0.5), "name": "Balanced-Visual"}
    ]

    for pid in tqdm(task_ids, desc="Processing Tasks"):
        meta = unified_meta[pid]
        target_b64 = get_image_b64_from_cos(cos_client, meta['cover_path'])
        if not target_b64: continue
        
        # 预计算 Task 的特征
        try:
            cos_obj = cos_client.get_object(Bucket=COS_CONFIG['Bucket'], Key=meta['cover_path'])
            img_raw = cos_obj['Body'].get_raw_stream().read()
            img_input = preprocess(Image.open(io.BytesIO(img_raw))).unsqueeze(0).to(device)
            text_input = clip.tokenize([meta['title']]).to(device)
            with torch.no_grad():
                i_feat = model.encode_image(img_input)
                i_feat /= i_feat.norm(dim=-1, keepdim=True)
                t_feat = model.encode_text(text_input)
                t_feat /= t_feat.norm(dim=-1, keepdim=True)
        except: continue

        for group in groups_config:
            # RAG 检索
            w_t, w_i = group["weight"]
            q_vec = (w_t * t_feat + w_i * i_feat)
            q_vec /= q_vec.norm(dim=-1, keepdim=True)
            D, I = group["index"].search(q_vec.cpu().numpy().astype('float32'), 4)
            
            # 构造多模态消息
            messages_payload = [
                {"type": "text", "text": "IMPORTANT: Below are 4 [REFERENCE EXAMPLES]. Use them only to understand the criteria. Do NOT label them."}
            ]
            
            for idx_rank, idx_val in enumerate(I[0]):
                ref_id = valid_ids_in_index[idx_val]
                ref_meta = unified_meta[ref_id]
                ref_b64 = get_image_b64_from_cos(cos_client, ref_meta['cover_path'])
                messages_payload.append({"type": "text", "text": f"[REFERENCE {idx_rank+1}] Title: {ref_meta['title']} | Known Labels: {ref_meta['label_info']}"})
                if ref_b64:
                    messages_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}", "detail": "low"}})

            messages_payload.append({"type": "text", "text": "--- END OF REFERENCES ---"})
            messages_payload.append({"type": "text", "text": f"NOW, evaluate the [TARGET SAMPLE] below for hooks: {group['hooks']}."})
            messages_payload.append({"type": "text", "text": f"[TARGET] Title: {meta['title']}"})
            messages_payload.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b64}", "detail": "low"}})

            total_calls += 1
            try:
                response = ai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": BASE_HOOK_DEFINITIONS + "\nInstruction: Distinguish between [REFERENCE] images and the [TARGET] image. Only output labels for the [TARGET]."},
                        {"role": "user", "content": messages_payload}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                res = json.loads(response.choices[0].message.content)
                for h_num in group["hooks"]:
                    final_decisions[pid][str(h_num)] = res.get(f"h{h_num}", 0)
            except: invalid_outputs += 1

    final_df = pd.DataFrame(list(final_decisions.values()))
    final_df.to_csv(RESULT_DIR / "final_machine_decision.csv", index=False, encoding='utf-8-sig')
    
    if os.path.exists(EXPERT_CSV):
        generate_validation_report(final_df, EXPERT_CSV)

if __name__ == "__main__":
    run_pipeline()