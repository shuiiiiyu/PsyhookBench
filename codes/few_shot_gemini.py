# -*- coding: utf-8 -*-
import json
import base64
import re
import os
import time
import csv
import random
import httpx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from qcloud_cos import CosConfig, CosS3Client

PRINT_INVALID_DETAIL = True
PRINT_EXCEPTION_DETAIL = True
PRINT_RAW_PREVIEW_ON_PARSE_FAIL = True
RAW_PREVIEW_CHARS = 600
WRITE_EVERY_ROW = True       
RESUME_IF_EXISTS = True         
SLEEP_PER_ROW_SEC = 1.0       
RATE_LIMIT_MAX_SLEEP = 60.0    
RATE_LIMIT_BASE_SLEEP = 2.0   
RATE_LIMIT_JITTER = 0.3       
RATE_LIMIT_MAX_RETRY = 20    

#路径与配置
WORKSPACE_DIR = Path("TEST")
TASK_CSV = "label_title_3005.csv"
TITLE_LOOKUP_CSV = str(WORKSPACE_DIR / "label_title_3005.csv")

RESULT_DIR = WORKSPACE_DIR / "ohmygpt_results"
RESULT_DIR.mkdir(exist_ok=True, parents=True)

#OhMyGPT 中转站配置
MODEL_NAME = "gemini-2.0-flash"   
BASE_URL = "https://api.ohmygpt.com/v1"
API_KEY = "sk-xxxx" 
# ===================== 2. 网络配置 =====================
http_client = httpx.Client(
    trust_env=True,
    timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0),
)
ai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

# ===================== 2.1 COS 配置 =====================
COS_CONFIG = {
    'SecretId': 'xxxx',
    'SecretKey': 'xxxx',
    'Region': 'xxxx',
    'Bucket': 'xxxx'
}
#3. 提示词与案例
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
核心定义：内容通过视觉或者文字将两种或多种形成反差意义的状态或事物放在一起，以激发受众的探索欲。
操作化判断：有没有两个及多个有明显反差意义的对比项，有比较但差别不大的不算
参照线索：
- 对比项包括：前后/左右/正反/好坏/预期与现实/别人与自己…等
- 对比形式包括：文本和文本的语义反差、图像之间的视觉反差、文本与图像之间的反差


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
内容中是否：明显的比较行为词 OR（展示差距 AND 展示了社会比较后态度）
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

[Json Output Format]
Return ONLY ONE valid JSON object (no markdown, no ```).
Keys MUST appear in this exact order. All h1~h8 MUST be 0 or 1.
"reasoning" MUST be Chinese and <= 30 characters. "reasoning" may be empty.

{
  "h1":0,
  "h2":0,
  "h3":0,
  "h4":0,
  "h5":0,
  "h6":0,
  "h7":0,
  "h8":0,
  "reasoning":"<=30字中文原因(可为空)"
}
"""

HOOK_CASES = {
    1: [
        ("64c4d22f0000000017019365", "Inaction (ignoring diet) leads to the cost of wasted workouts, triggering FOMO.", 1),
        ("69706f2e0000000022030f10", "Failing to choose the right fridge causes health risks, fitting FOMO's inaction-cost logic.", 1),
        ("63f8c1d20000000012032891", "No inaction-cost link; just personal avoidance, not FOMO-inducing loss fear.", 0),
    ],
    2: [
        ("68f0b6ee000000000400330b", "Offers styling skills to look better and avoid a plain appearance", 1),
        ("6960f3ea00000000210284d0", "Provides curated autumn routes to save time and enjoy fall views efficiently", 1),
        ("640d649400000000130087a1", "Only shows bedroom design with no stated benefits for the audience", 0),
    ],
    3: [
        ("696f532b000000001a037aee", "Truncates with ellipsis after 'unless', intentionally hiding key conditions to drive clicks", 1),
        ("690ca1f5000000000700e328", "Withholds the suspense drama’s name, creating a core info gap to lure clicks", 1),
        ("6440c966000000001300a8cc", "No deliberate hiding; title fully conveys core Meta layoff news clearly", 0),
    ],
    4: [
        ("63fc10dd0000000013010d59", "Author uses '吓一跳' to frame ¥100 skewers as astonishingly abnormal.", 1),
        ("68ecbd86000000000303b778", "Author uses '封神级别' to package Hokkaido winter as extremely rare/novel.", 1),
        ("644a7b200000000013002c23", "No astonishing/rare packaging; normal sharing about cute bags.", 0),
    ],
    5: [
        ("64ad5371000000001c00dac1", "Shows stark visual contrast between day1 and day3 belly states.", 1),
        ("64565f230000000011010f20", "Contrasts DIY cost ($200+) with claimed $1799 value sharply.", 1),
        ("6969de13000000002102ab83", "Lipstick shades differ slightly; no significant contrast.", 0),
    ],
    6: [
        ("696c839b0000000022021da9", "Mentions 'Chinese women(中女)' as group label + ingroup belonging/affirmation.", 1),
        ("6965bc51000000000d009ec1", "Uses 'Li Zeyan's fans' group label + calls for group attention/action.", 1),
        ("6445065300000000120334a7", "Lacks a clear group label or ingroup/outgroup attitude.", 0),
    ],
    7: [
        ("690c6efa000000000400119d", "Uses 'not inferior to' to compare targets in status/achievement.", 1),
        ("6470075400000000270118bc", "Shows gap in figure/appearance beyond average + attitude to improve.", 1),
        ("641c0cd5000000001303ee42", "Only states loneliness as normal; no comparison/gap/attitude.", 0),
    ],
    8: [
        ("6429995c0000000013030a41", "CCTV endorsement acts as authority to reduce audience doubt.", 1),
        ("643ba7940000000012032cb1", "TV station feature serves as authority endorsement for credibility.", 1),
        ("69677b67000000001a02e15e", "No real authoritative source; only subjective 'authority' claim.", 0),
    ],
}

#工具函数
def get_cos_client():
    if not COS_CONFIG["SecretId"] or not COS_CONFIG["SecretKey"]:
        raise RuntimeError("请设置 COS_SECRET_ID / COS_SECRET_KEY 环境变量（或写死在 COS_CONFIG 里）")
    config = CosConfig(
        Region=COS_CONFIG["Region"],
        SecretId=COS_CONFIG["SecretId"],
        SecretKey=COS_CONFIG["SecretKey"]
    )
    return CosS3Client(config)

def load_title_map(label_title_csv: str) -> dict:
    df = pd.read_csv(label_title_csv, dtype=str, keep_default_na=False)
    df["post_id"] = df["post_id"].astype(str).str.strip().str.lower()
    df["title"] = df["title"].astype(str)
    return dict(zip(df["post_id"], df["title"]))

def build_cases_block(title_map: dict) -> str:
    lines = ["\n\n[Calibration Cases]\n", "Each hook below has 3 cases. The 'label' refers ONLY to that hook.\n"]
    for hook_id in range(1, 9):
        lines.append(f"\n## hook{hook_id} cases\n")
        for j, (pid, reasoning, label) in enumerate(HOOK_CASES[hook_id], start=1):
            pid_l = pid.strip().lower()
            title = title_map.get(pid_l, "")
            lines.append(f"- case{j}: post_id={pid_l}\n  title={title}\n  reasoning={reasoning}\n  label={label}\n")
    return "\n".join(lines)

def detect_image_mime(data: bytes) -> str:
    # JPEG
    if len(data) >= 3 and data[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # PNG
    if len(data) >= 8 and data[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WEBP (RIFF....WEBP)
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # GIF
    if len(data) >= 6 and data[0:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"

def fetch_cover_b64_and_mime(cos_client, pid: str):
    key = f"downloads/{pid}/cover.jpg"
    obj = cos_client.get_object(Bucket=COS_CONFIG["Bucket"], Key=key)
    data = obj["Body"].get_raw_stream().read()
    mime = detect_image_mime(data)
    b64 = base64.b64encode(data).decode("utf-8")
    return b64, mime

def safe_parse_json(text: str):
    if not text:
        return None
    s = text.strip()

    s = re.sub(r"```json\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")

    # 从第一个 { 开始做括号配对，截取第一个完整 JSON 对象
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        return json.loads(candidate)
                    except:
                        return None
    return None

def normalize_01(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v) if v in (0, 1) else None
    if isinstance(v, str):
        t = v.strip()
        if t == "0":
            return 0
        if t == "1":
            return 1
    return None

def validate_and_extract(raw_text: str):
    obj = safe_parse_json(raw_text)
    if not isinstance(obj, dict):
        return False, None, "JSON_PARSE_FAILED"

    # 额外机制键检查：只要出现 h9/h10 这种，直接判 invalid
    for k in obj.keys():
        m = re.match(r"^(h|hook)(\d+)$", str(k).strip())
        if m and (int(m.group(2)) < 1 or int(m.group(2)) > 8):
            return False, None, f"EXTRA_HOOK_KEY_{k}"

    extracted = {}

    # h1~h8 必须可读为 0/1
    for i in range(1, 9):
        val = normalize_01(obj.get(f"h{i}"))
        if val is None:
            return False, None, f"INVALID_VALUE_h{i}"
        extracted[f"h{i}"] = val

    # reasoning 允许为空：缺失/空串都 OK，最后兜底截断 20 字
    r = str(obj.get("reasoning", "") if obj.get("reasoning", "") is not None else "").strip()
    extracted["reasoning"] = r[:20]

    return True, extracted, ""

def _short(s: str, n: int):
    s = "" if s is None else str(s)
    s = s.replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")

def looks_truncated_json(s: str) -> bool:
    if not s:
        return True
    s = s.strip()
    if "{" in s and "}" not in s:
        return True

    start = s.find("{")
    if start == -1:
        return False

    depth = 0
    in_str = False
    esc = False
    for ch in s[start:]:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return False
    return True

def is_rate_limited_exception(e: Exception) -> bool:
    msg = str(e).lower()
    keywords = [
        "rate limit", "ratelimit", "too many requests", "429",
        "quota", "throttl", "exceeded", "temporarily unavailable"
    ]
    if any(k in msg for k in keywords):
        return True
    if hasattr(e, "status_code") and getattr(e, "status_code", None) == 429:
        return True
    if hasattr(e, "response") and getattr(e, "response", None) is not None:
        try:
            if getattr(e.response, "status_code", None) == 429:
                return True
        except Exception:
            pass
    return False

def format_exception_detail(e: Exception) -> str:
    parts = [f"{type(e).__name__}: {e}"]
    if hasattr(e, "status_code") and getattr(e, "status_code", None):
        parts.append(f"status_code={getattr(e, 'status_code')}")
    if hasattr(e, "body") and getattr(e, "body", None):
        parts.append(f"body={_short(getattr(e, 'body'), 800)}")
    if hasattr(e, "response") and getattr(e, "response", None):
        try:
            r = getattr(e, "response")
            if hasattr(r, "status_code"):
                parts.append(f"resp_status={r.status_code}")
        except Exception:
            pass
    return " | ".join(parts)

#写入 CSV
def load_done_post_ids(output_csv: Path) -> set:
    if (not output_csv.exists()) or output_csv.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=["post_id"], keep_default_na=False)
        return set(df["post_id"].astype(str).str.strip().str.lower().tolist())
    except Exception:
        return set()

def open_csv_appender(output_csv: Path, fieldnames: list):
    need_header = (not output_csv.exists()) or output_csv.stat().st_size == 0
    f = open(output_csv, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if need_header:
        writer.writeheader()
        f.flush()
    return f, writer

def write_one_row(writer, f_handle, rec: dict):
    writer.writerow(rec)
    f_handle.flush()

#主流程
def run_test():
    title_map = load_title_map(TITLE_LOOKUP_CSV)
    cases_block = build_cases_block(title_map)
    system_prompt = BASE_HOOK_DEFINITIONS + cases_block

    df_task = pd.read_csv(TASK_CSV, dtype=str, keep_default_na=False)
    df_task["post_id"] = df_task["post_id"].astype(str).str.strip().str.lower()

    cos_client = get_cos_client()

    OUT_COLS = ["post_id"] + [f"h{i}" for i in range(1, 9)] + ["title", "reasoning"]
    output_path = RESULT_DIR / f"predictions_{MODEL_NAME.replace('/', '_')}_incremental.csv"

    done = set()
    if RESUME_IF_EXISTS:
        done = load_done_post_ids(output_path)
        if done:
            print(f"🔁 RESUME: 检测到已完成 {len(done)} 条，将自动跳过。输出：{output_path}")

    f_out, writer = open_csv_appender(output_path, OUT_COLS)

    total_model_calls = 0
    invalid_model_calls = 0

    try:
        for _, row in tqdm(df_task.iterrows(), total=len(df_task), desc=f"Running {MODEL_NAME}"):
            pid = row["post_id"]
            if RESUME_IF_EXISTS and pid in done:
                continue

            title = title_map.get(pid, "")

            try:
                img_b64, img_mime = fetch_cover_b64_and_mime(cos_client, pid)
            except Exception as e:
                if PRINT_EXCEPTION_DETAIL:
                    tqdm.write(f"[COS_ERROR] pid={pid} | {format_exception_detail(e)}")
                rec = {
                    "post_id": pid,
                    "title": title,
                    "reasoning": "MISSING_COVER_OR_COS_ERROR",
                    **{f"h{i}": 0 for i in range(1, 9)}
                }
                if WRITE_EVERY_ROW:
                    write_one_row(writer, f_out, rec)
                done.add(pid)
                if SLEEP_PER_ROW_SEC > 0:
                    time.sleep(SLEEP_PER_ROW_SEC)
                continue

            final = None
            last_err = ""

            invalid_attempts = 0  
            rate_retry = 0       
            backoff = RATE_LIMIT_BASE_SLEEP

            while invalid_attempts < 3:
                total_model_calls += 1
                try:
                    resp = ai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": f"待标注标题: {title}"},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:{img_mime};base64,{img_b64}"}}
                            ]}
                        ],
                        temperature=0,
                        max_tokens=200, 
                    )

                    raw_content = resp.choices[0].message.content
                    ok, extracted, err = validate_and_extract(raw_content)

                    if ok:
                        final = extracted
                        break

                    if err == "JSON_PARSE_FAILED" and looks_truncated_json(raw_content):
                        rate_retry += 1
                        if PRINT_INVALID_DETAIL:
                            tqdm.write(
                                f"[TRUNCATED_JSON] pid={pid} | retry={rate_retry}/{RATE_LIMIT_MAX_RETRY} "
                                f"| len={len(raw_content)} | tail={repr(raw_content[-80:])}"
                            )
                        if PRINT_RAW_PREVIEW_ON_PARSE_FAIL:
                            tqdm.write(f"[RAW_PREVIEW] pid={pid} | {_short(raw_content, RAW_PREVIEW_CHARS)}")

                        if rate_retry > RATE_LIMIT_MAX_RETRY:
                            last_err = "TRUNCATED_JSON_MAX_RETRY_EXCEEDED"
                            break

                        sleep_s = min(RATE_LIMIT_MAX_SLEEP, backoff)
                        jitter = 1.0 + random.uniform(-RATE_LIMIT_JITTER, RATE_LIMIT_JITTER)
                        sleep_s = max(0.5, sleep_s * jitter)
                        tqdm.write(f"[TRUNCATED_SLEEP] pid={pid} | sleep={sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        backoff *= 2.0
                        continue

                    invalid_model_calls += 1
                    invalid_attempts += 1
                    last_err = err

                    if PRINT_INVALID_DETAIL:
                        tqdm.write(f"[INVALID_OUTPUT] pid={pid} | invalid_try={invalid_attempts}/3 | err={err}")
                        if err == "JSON_PARSE_FAILED" and PRINT_RAW_PREVIEW_ON_PARSE_FAIL:
                            tqdm.write(f"[RAW_PREVIEW] pid={pid} | {_short(raw_content, RAW_PREVIEW_CHARS)}")

                    time.sleep(0.3)

                except Exception as e:
                    if is_rate_limited_exception(e):
                        rate_retry += 1
                        if PRINT_EXCEPTION_DETAIL:
                            tqdm.write(
                                f"[RATE_LIMIT] pid={pid} | retry={rate_retry}/{RATE_LIMIT_MAX_RETRY} | "
                                f"{format_exception_detail(e)}"
                            )
                        if rate_retry > RATE_LIMIT_MAX_RETRY:
                            last_err = "RATE_LIMIT_MAX_RETRY_EXCEEDED"
                            break

                        sleep_s = min(RATE_LIMIT_MAX_SLEEP, backoff)
                        jitter = 1.0 + random.uniform(-RATE_LIMIT_JITTER, RATE_LIMIT_JITTER)
                        sleep_s = max(0.5, sleep_s * jitter)

                        tqdm.write(f"[RATE_LIMIT_SLEEP] pid={pid} | sleep={sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        backoff *= 2.0
                        continue

                    invalid_model_calls += 1
                    invalid_attempts += 1
                    last_err = f"API_EXCEPTION_{type(e).__name__}"
                    if PRINT_EXCEPTION_DETAIL:
                        tqdm.write(f"[API_EXCEPTION] pid={pid} | invalid_try={invalid_attempts}/3 | {format_exception_detail(e)}")
                    time.sleep(0.8)

            if final is None:
                tqdm.write(f"[FINAL_FAIL] pid={pid} | last_err={last_err}")
                rec = {
                    "post_id": pid,
                    "title": title,
                    "reasoning": f"FAILED:{last_err}"[:20],  
                    **{f"h{i}": 0 for i in range(1, 9)}
                }
            else:
                r = (final.get("reasoning", "") or "").strip()[:20]  
                rec = {
                    "post_id": pid,
                    "title": title,
                    "reasoning": r,
                    **{f"h{i}": final[f"h{i}"] for i in range(1, 9)}
                }

            if WRITE_EVERY_ROW:
                write_one_row(writer, f_out, rec)

            done.add(pid)

            if SLEEP_PER_ROW_SEC > 0:
                time.sleep(SLEEP_PER_ROW_SEC)

    finally:
        f_out.close()

    invalid_rate = (invalid_model_calls / total_model_calls) if total_model_calls > 0 else 0.0
    stats = {
        "model_name": MODEL_NAME,
        "task_csv": TASK_CSV,
        "output_csv": str(output_path),
        "total_model_calls": total_model_calls,
        "invalid_model_calls": invalid_model_calls,
        "invalid_rate": f"{invalid_rate:.4%}",
        "processed_rows": len(done),
        "sleep_per_row_sec": SLEEP_PER_ROW_SEC,
        "rate_limit_base_sleep": RATE_LIMIT_BASE_SLEEP,
        "rate_limit_max_sleep": RATE_LIMIT_MAX_SLEEP,
        "rate_limit_max_retry": RATE_LIMIT_MAX_RETRY,
    }
    with open(RESULT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n完成！输出：{output_path}")
    print(f"调用总数 {total_model_calls} | 无效输出 {invalid_model_calls} | 无效率 {invalid_rate:.4%} | 已写入 {len(done)} 行")

if __name__ == "__main__":
    run_test()
