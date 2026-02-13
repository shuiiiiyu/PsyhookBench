import os
import json
import base64
import httpx
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
from openai import OpenAI

# ===================== 1. 路径与配置 =====================
TEST_CSV = "2500.csv"
REF_CSV  = "541_raw_data.csv"
DOWNLOADS_DIR = "downloads"

os.makedirs("ethics", exist_ok=True)
OUT_CSV  = "ethics/ethics_out.csv"
OUT_JSONL = "ethics/ethics_out.jsonl"

MODEL_NAME = "qwen-vl-max" 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-xxxx" 

# ===================== 2. 网络配置 =====================
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com"
http_client = httpx.Client(trust_env=False, proxies=None, timeout=60.0)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

# ===================== 3. 媒体处理函数 (优化版) =====================
MAX_IMG_SIDE = 1024
JPEG_QUALITY = 80 

def _file_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _compress_img_to_b64(img_path: Path) -> str:
    """压缩图片以节省 Token 消耗"""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = min(MAX_IMG_SIDE / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _sample_video_frames(video_path: Path, n_frames: int = 4) -> List[str]:
    """对视频进行抽帧，这是处理视频最省钱的方式"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count < n_frames: n_frames = max(1, frame_count)
    
    idxs = [int(i * (frame_count - 1) / (n_frames - 1)) if n_frames > 1 else 0 for i in range(n_frames)]
    frames_b64 = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    cap.release()
    return frames_b64

# ===================== 4. 核心逻辑 =====================
def build_smart_content(pid: str, text: str) -> List[Dict]:
    """
    智能选择模态：
    1. 图片文件使用 image_url (按像素计费)
    2. 视频文件使用 video (抽帧处理，避免传输超大文件)
    """
    folder = Path(DOWNLOADS_DIR) / pid
    content_list = []
    
    if not folder.exists():
        return [{"type": "text", "text": text}]

    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    vid_exts = {".mp4", ".mov", ".avi"}
    
    all_files = sorted(list(folder.iterdir()))
    
    images_b64 = []
    cover = folder / "cover.jpg"
    if cover.exists():
        images_b64.append(_compress_img_to_b64(cover))
    
    for f in all_files:
        if f.suffix.lower() in img_exts and f.name != "cover.jpg":
            images_b64.append(_compress_img_to_b64(f))
    
    # 将图片以 image_url 形式加入
    for b64 in images_b64[:8]: 
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # 处理视频 (Video 抽帧模式)
    video_frames_b64 = []
    for f in all_files:
        if f.suffix.lower() in vid_exts:
            video_frames_b64.extend(_sample_video_frames(f, n_frames=4))
    
    if video_frames_b64:
        # 视频帧打包成 video 类型
        video_data = [f"data:image/jpeg;base64,{b}" for b in video_frames_b64]
        content_list.append({"type": "video", "video": video_data})

    content_list.append({"type": "text", "text": text})
    return content_list

# ===================== 5. 准则与主逻辑 =====================
RUBRIC_TEXT = """你是内容审核员。给你一条小红书内容卡片：标题、封面、展开正文、以及该内容的所有图片/视频帧（封面在第一张）,其中预览内容只封面和标题，预览内容不包括展开正文和除封面外所有图片视频。
请先阅读预览内容，再把预览内容和正文内容、封面后所有图片视频呈现的内容对比，判断是否存在以下伦理风险，可多选，并输出严格JSON（不要多余文字）：
{
  "a_missing": 0/1,   // 缺失：预览内容通过明确的价值承诺、信息指向或内容预告（如干货、教程、事件解答、问题解决办法、具体体验分享等），向用户传递可预期的核心内容；用户点击展开后，预览内容所承诺的核心内容完全未呈现；比如封面上的主体物视频中没有；但是排除小红书个人分享类内容中，装饰性文案与自拍、日常随拍等个人化配图弱关联的正常表达情形。
  "b_wrong": 0/1,     // 错误：预览内容所表达的核心信息、观点、事实与展开后的内容主体存在实质性偏差、矛盾或歪曲；排除小红书个人分享类内容中，为营造分享氛围，使用的装饰性文案与自拍、日常随拍等个人化配图无强关联的正常表达情形。
  "c_incitement": 0/1,// 煽动：预览内容使用不恰当、粗俗的词语，过度诱导用户点击，违背平台内容表达的文明规范。
  "d_bad": 0/1,       // 不良影响：预览内容宣传违反国家法律法规、社会公序良俗、平台社区规则的内容，或传递低俗、暴力、违法、违背道德的信息，易对用户产生不良价值引导。
  "e_none": 0/1       // 无伦理风险：当且仅当 a/b/c/d 全为0
}
"""

REF_POSTS = [
    ("690c7fb40000000005013c77", {"a_missing": 1, "b_wrong": 0, "c_incitement": 0, "d_bad": 0, "e_none": 0}, "参考：缺失"),
    ("690df79c0000000003037217", {"a_missing": 1, "b_wrong": 0, "c_incitement": 0, "d_bad": 1, "e_none": 0}, "参考：擦边诱导+缺失（按不良影响处理）"),
    ("68fc904d0000000007037025", {"a_missing": 1, "b_wrong": 0, "c_incitement": 0, "d_bad": 0, "e_none": 0}, "参考：缺失"),
    ("692a8fac000000001d03dead", {"a_missing": 0, "b_wrong": 0, "c_incitement": 0, "d_bad": 1, "e_none": 0}, "参考：题材敏感（按不良影响处理）"),
]

def build_fewshot(ref_df: pd.DataFrame) -> List[Dict]:
    msgs = []
    for pid, label, note in REF_POSTS: 
        match = ref_df.loc[ref_df["post_id"].astype(str) == pid]
        if match.empty: continue
        row = match.iloc[0]
        text = f"参考案例 | 标题：{row['title']}\n正文：{row['content']}"
        msgs.append({"role": "user", "content": build_smart_content(pid, text)})
        msgs.append({"role": "assistant", "content": json.dumps(label)})
    return msgs

def main():
    test_df = pd.read_csv(TEST_CSV)
    ref_df = pd.read_csv(REF_CSV)
    
    print("构建 Few-shot 环境...")
    fewshot = build_fewshot(ref_df)
    
    done = set()
    if Path(OUT_CSV).exists():
        try: done = set(pd.read_csv(OUT_CSV)["post_id"].astype(str).tolist())
        except: pass

    jsonl_f = open(OUT_JSONL, "a", encoding="utf-8")
    
    for _, r in tqdm(test_df.iterrows(), total=len(test_df)):
        pid = str(r["post_id"])
        if pid in done: continue
        
        try:
            prompt = f"{RUBRIC_TEXT}\n待审：\n标题：{r.get('title','')}\n正文：{r.get('content','')}"
            curr_content = build_smart_content(pid, prompt)
            
            resp = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=fewshot + [{"role": "user", "content": curr_content}],
                temperature=0.0
            )
            txt = resp.choices[0].message.content
            
            # 解析 JSON
            l, r_idx = txt.find("{"), txt.rfind("}")
            res = json.loads(txt[l:r_idx+1])
            res["post_id"] = pid
            
            # 存盘
            jsonl_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            jsonl_f.flush()
            pd.DataFrame([res]).to_csv(OUT_CSV, mode='a', index=False, header=not Path(OUT_CSV).exists())
            
        except Exception as e:
            print(f"ID {pid} 失败: {e}")

    jsonl_f.close()
    print("任务完成。")

if __name__ == "__main__":
    main()
