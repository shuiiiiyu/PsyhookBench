from flask import (
    Flask, render_template, request,
    redirect, url_for, abort, session, send_file
)
import json
import pandas as pd
import re, os
from qcloud_cos import CosConfig, CosS3Client
from io import StringIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ========= COS 配置 (保持原样) =========
SECRET_ID = 'xxxx'
SECRET_KEY = 'xxxx'
REGION = 'xxxx'
BUCKET_NAME = 'xxxx'

CSV_FILES = {
    "伦理风险391条": "xxx",
    "第一次标注": "xxx",
    "第二次标注":"xxx",
    #"补充的机制":"keyword_1.csv"
}

config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
client = CosS3Client(config)

# --- 辅助函数 ---
def get_current_csv_key():
    csv_label = session.get('csv_label')
    return CSV_FILES.get(csv_label, list(CSV_FILES.values())[0])

def get_csv_from_cos():
    try:
        csv_key = get_current_csv_key()
        response = client.get_object(Bucket=BUCKET_NAME, Key=csv_key)
        csv_data = response['Body'].get_raw_stream().read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        df['post_id'] = df['post_id'].astype(str)
        return df
    except: return None

def get_post_ids_from_csv():
    df = get_csv_from_cos()
    return df['post_id'].tolist() if df is not None else []

def get_post_detail(post_id):
    df = get_csv_from_cos()
    if df is None: return "No Title", ""
    row = df[df['post_id'] == post_id]
    if row.empty: return "No Title", ""
    return str(row.iloc[0].get('title', 'No Title')), str(row.iloc[0].get('content', ''))

# ========= 选项定义 (仅修正语法错误) =========
question_types = [
    {
        "id": "1",
        "title": "Fear Of Missing Out (FOMO)",
        "definition": 'The content expresses "if you don\'t do something, there will be certain consequences (losses / regrets)" to arouse the audience\'s concerns and increase tension.',
        "details": """<strong>Operational Judgment:\nDoes the content contain/convey: clues of "inaction" AND "costs" or "consequences" of "inaction"</strong>
<strong>Reference Clues:</strong>
 <strong>• Clues of "inaction":</strong> words like not watching / not listening / not doing / not acting like what is mentioned in the post...
 <strong>• "Costs" and "consequences" of "inaction" include:</strong> negative intentions such as bankruptcy / breakup / failure..."""
    },
    {
        "id": "2",
        "title": "Gain Appeal",
        "definition": 'The content emphasizes "what benefits can be obtained through this content information" to stimulate the audience\'s inherent motivation to acquire.',
        "details": """<strong>Operational Judgment:\nDoes the content contain/convey: the benefits that the content can bring</strong>
<strong>Reference Clues:</strong>
 <strong>• Benefits:</strong> Money (saving money, making money), time (saving time, improving efficiency), health (losing weight, becoming more beautiful), skills (quick learning, mastery), emotions (happiness, peace of mind)... and other positive intentions"""
    },
    {
        "id": "3",
        "title": "Information-gap",
        "definition": "When the content could have fully conveyed information through the title or cover, it intentionally omits part of the information to guide the audience to click and find it.",
        "details": """<strong>Operational Judgment:\nDoes the content create: intentionally hidden information</strong>
<strong>Reference Clues:</strong>
 <strong>• Self-questioning and self-answering:</strong> Raising a question in the title or cover and answering it in the content
 <strong>• Obscuring key information:</strong> Using mosaics, stickers, etc., to cover key parts of the title or cover
 <strong>• Leaving words half-said:</strong> Truncating sentences or stories with ellipsis, interruptions, leaving blank space, etc.
 <strong>• Setting suspense:</strong> Using various forms to render the missing information
 <strong>• Only presenting scenarios:</strong> For example, "When...", "pov:..."
 <strong>• Ambiguous reference:</strong> Using words like "this" or "that" to refer to something, but it is unclear what exactly they refer to
（Or: clues not listed above but conforming to the core definition）"""
    },
    {
        "id": "4",
        "title": "Anomaly and novelty",
        "definition": "Content is deliberately packaged to be astonishing, counterintuitive, rare, or novel in order to arouse the audience's curiosity.",
        "details": """<strong>Operational Judgment:\nDoes the content contain: phrases that express astonishing abnormality OR phrases that express rarity and novelty</strong>
<strong>Reference Clues:</strong>
 <strong>• Phrases expressing astonishing abnormality:</strong> actually/unexpectedly (竟然), unexpectedly (居然), didn't expect (没想到), incredible (不可思议), rare (罕见), seeing for the first time (第一次见), shocked (震惊), stunned (惊呆), dumbfounded (看傻), refreshing one's outlook on life (刷新三观), refreshing one's cognition (刷新认知), incredible (不可思议), magical (神奇)…
 <strong>• Phrases expressing rarity and novelty include:</strong>- Words directly claiming novelty: original (独创), unique (别具一格), novel and different (标新立异), novel (新颖)…
                                                - Extreme words: most (最), top (顶), super (超), first (第一), epic-level (史诗级)…
                                                - Scarcity: only (唯一), only (只有), limited (限定), rarely seen (鲜见), occasional (偶发), unique case/unique item (孤例孤品), niche (小众), unpopular (冷门), once in a millennium (千年一遇)…
(Note: Regardless of the content itself, it counts as such as long as it is packaged to be abnormal or novel; conversely, if the content itself is abnormal or novel enough but not packaged by the author, it does not count.)"""
    },
    {
        "id": "5",
        "title": "Perceptual Contrast",
        "definition": "Content places two or more contrasting states or things together visually or textually to stimulate the audience's desire to explore.",
        "details": """<strong>Operational Judgment:\nDoes the content contain/convey: two or more contrasting items with obvious differences</strong>
<strong>Reference Clues:</strong>
<strong> • Contrasting items include:</strong> front and back/left and right/positive and negative/good and bad/expectation and reality/others and oneself... etc.
<strong> • Forms of contrast include:</strong> semantic contrast between texts, visual contrast between images, contrast between text and image
<strong>Note:</strong> It is necessary to identify and perceive any clues that may constitute a contrast"""
    },
    {
        "id": "6",
        "title": "Ingroup Identification / Outgroup Distinction",
        "definition": "Content uses group labels to arouse a sense of identity and belonging within a certain group; or to arouse rejection and ridicule towards a certain group.",
        "details": """<strong>Operational Judgment:\nIn the content: Does it contain group labels AND (display attitudes of belonging/rejection OR calls to action OR group commonalities)</strong>
<strong>Reference Clues:</strong>
 <strong>• Group Labels:</strong> Mention of a certain group, such as
                    - Mention: nouns related to race/country/ethnicity/religion/region...
                    - Mention: nouns related to school/company/organization/institution/
                    - Mentioned: nouns related to age/gender/occupation... etc.
                    - And others: certain hobby tags/certain personality tags/certain zodiac signs/certain MBTI tags... etc.
          （As long as the appearing phrase can divide a group from others in the crowd, it is acceptable）
 <strong>• Attitude of belonging/exclusion:</strong> Showing pride, honor, identification, empathy...; or: disdain, teasing, satire... etc.
 <strong>• Call to action:</strong> "...must watch", "like if you are...", "...let's take action"... etc.
 <strong>• Group commonalities:</strong> "Every... has experienced it", "the daily life of...", "...all understand"... and common characteristics reflected in the pictures, etc."""
    },
    {
        "id": "7",
        "title": "Social Comparison",
        "definition": "Content triggers the audience to engage in comparison by directly using obvious comparative words, showing gaps and displaying certain attitudes after social comparison.",
        "details": """<strong>Operational Judgment:\nIn the content: Are there obvious comparative action words OR (display of gaps AND display of attitudes after social comparison)</strong>
<strong>Reference Clues:</strong>
 <strong>• Obvious comparative action words:</strong> than... (比...), more (更), VS, not as good as... (不如...), crush (碾压), instantly defeat... (秒杀…) etc.
 <strong>• Gaps:</strong> Ability gaps (skills, achievements, task performance, etc.) / Self-trait gaps (appearance, personality, talent, height, etc.) / Resource gaps (wealth, living standards, status, power differences, etc.)...
 <strong>• Attitudes after comparison:</strong>- Upward comparison: Negative attitudes such as jealousy and inferiority; Positive attitudes such as recognition and motivation to make progress.
                            - Downward comparison: Negative attitudes such as showing off and a sense of superiority; Positive attitudes such as cherishing and being content.
<strong>Note:</strong> The reference point for the gap can be explicit (e.g., a direct object is shown) or implicit (e.g., the generally accepted level by the public)."""
    },
    {
        "id": "8",
        "title": "Authority Endorsement",
        "definition": "Content uses various persuasive source endorsements to exempt the audience from the cost of questioning, thereby guiding the audience to be convinced or imitate.",
        "details": """<strong>Operational Judgment:\nDoes the content contain: source endorsement</strong>
<strong>Reference Clues:</strong>
 <strong>• Authoritative sources include:</strong> experts/professors/institutions/celebrities/research/rankings/certifications/awards/numbers... etc.
    - For example: Harvard University research/According to... reports/... experts say/FDA certification/XX celebrity's same style/Qingdao's top 1…/220,000 people have viewed…"""
    }
]


extra_options = [
    {"id": "E1", "title": "Missing", "definition": "The content promised by the title is missing after expansion."},
    {"id": "E2", "title": "Incorrect", "definition": "The title does not match the actual content."},
    {"id": "E3", "title": "Inflammatory / Vulgar", "definition": "Uses inappropriate, offensive, or vulgar wording."},
    {"id": "E4", "title": "Harmful Impact", "definition": "Promotes content that violates laws or moral norms."},
    {"id": "E5", "title": "No Ethical Risk", "definition": ""}
]

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form.get('username', '').strip()
        session['csv_label'] = request.form.get('csv_label', '')
        return redirect(url_for('index'))
    return render_template('login.html', csv_files=CSV_FILES)

@app.route('/media/<post_id>/<filename>')
def serve_media(post_id, filename):
    file_key = f'downloads/{post_id}/{filename}'
    try:
        response = client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        mimetype = 'video/mp4' if filename.lower().endswith('.mp4') else 'image/jpeg'
        return send_file(response['Body'].get_raw_stream(), mimetype=mimetype)
    except: abort(404)

@app.route('/')
def index():
    if 'username' not in session: return redirect(url_for('login'))
    return redirect(url_for('view_item', idx=0))

@app.route('/item/<int:idx>')
def view_item(idx):
    if 'username' not in session: return redirect(url_for('login'))
    post_ids = get_post_ids_from_csv()
    if not post_ids or idx >= len(post_ids): abort(404)
    
    post_id = post_ids[idx]
    title, content = get_post_detail(post_id)
    
    media_list = []
    try:
        resp = client.list_objects(Bucket=BUCKET_NAME, Prefix=f'downloads/{post_id}/')
        if 'Contents' in resp:
            media_list = [obj['Key'].split('/')[-1] for obj in resp['Contents'] 
                         if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
            media_list.sort()
    except: media_list = ['cover.jpg']

    username = session.get('username')
    file_path = f"{username}.json"
    annotations = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                annotations = json.load(f)
            except: annotations = {}
    
    # 获取存过的数据
    post_data = annotations.get(post_id, {})
    # 这里的 key 必须和 save_and_next 存入的时候一致
    saved_hooks = post_data.get('hook_scores', {})
    saved_ethics = post_data.get('ethics_scores', {})

    return render_template(
        'item1.html',
        post_id=post_id, title=title, content=content,
        media_list=media_list,
        options=question_types,
        extra_options=extra_options,
        saved_hooks=saved_hooks,
        saved_ethics=saved_ethics,
        idx=idx, progress=idx+1, total_images=len(post_ids),
        is_last_image=(idx == len(post_ids)-1)
    )

@app.route('/save_and_next', methods=['POST'])
def save_and_next():
    username = session.get('username')
    file_path = f"{username}.json"
    post_id = request.form['post_id']
    idx = int(request.form['idx'])

    hook_scores = {}
    ethics_scores = {}

    for key, value in request.form.items():
        if key.startswith('score_'):
            val = int(value)
            opt_id = key.replace('score_', '')
            if opt_id.startswith('E'):
                ethics_scores[opt_id] = val
            else:
                hook_scores[opt_id] = val

    annotations = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try: annotations = json.load(f)
            except: annotations = {}
            
    annotations[post_id] = {
        'title': request.form.get('title_hidden', ''),
        'hook_scores': hook_scores,
        'ethics_scores': ethics_scores
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)
    
    post_ids = get_post_ids_from_csv()
    return redirect(url_for('view_item', idx=min(idx + 1, len(post_ids)-1)))
    
@app.route('/jump_to', methods=['POST'])
def jump_to():
    target = int(request.form.get('target_idx', 1)) - 1
    return redirect(url_for('view_item', idx=target))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)