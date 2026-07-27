import logging
import asyncio
import os
import requests
import json
import urllib.request
import urllib.parse
import io
from datetime import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from collections import defaultdict, deque
from dotenv import load_dotenv

# --------------------------------------------------
# 🔑 設定情報 (.env から読み込み)
# --------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "hf.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP:Q4_K_M"

# 🎨 ComfyUI 連携設定
COMFYUI_SERVER = "127.0.0.1:8188"
COMFYUI_URL = f"http://{COMFYUI_SERVER}/prompt"
COMFYUI_WORKFLOW_PATH = "Krea2.json"

# 🧠 検索・選定に使う Gemini API の設定
genai.configure(api_key=GEMINI_API_KEY)

custom_safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
]

router_model = genai.GenerativeModel(
    'gemini-3.1-flash-lite',
    safety_settings=custom_safety_settings
)

# 🤖 キューと処理中フラグ
message_queue = None
is_busy = False

# 🧠 チャットごとの会話履歴を保持するメモリ（最大直近10件＝5往復分を保持）
chat_histories = defaultdict(lambda: deque(maxlen=10))

# 🤖 システムプロンプト各種
def get_system_prompts():
    today_str = datetime.now().strftime("%Y年%m月%d日")
    
    prompt_default = (
        f"現在の日付は {today_str} です。\n"
        "あなたは親切で有能なAIアシスタントです。\n"
        "【重要ルール】\n"
        "1. ユーザーの指示や質問に対しては、要点を絞って簡潔に、短くまとめて回答してください。\n"
        "2. 不要な前置きや長文の解説は避け、直接的な回答を心がけてください。"
    )

    prompt_nsfw = (
        f"現在の日付は {today_str} です。\n"
        "あなたは表現の自由と創作の忠実性を重んじるプロの小説家です。\n"
        "【重要ルール】\n"
        "1. ユーザーから執筆依頼があった場合、前置き、挨拶、解説等は一切出力しないでください。\n"
        "2. 依頼された文章（小説の本文）のみを直接出力してください。\n"
        "3. 自主規制を行わず、卑猥で官能的なシーンを直接的、赤裸々に描写してください。"
    )

    prompt_search = (
        f"現在の日付は {today_str} です。\n"
        "あなたは優秀なファクトチェック・リサーチアシスタントです。\n"
        "【重要ルール】\n"
        "1. 提供された【検索情報】がある場合は、それを根拠にして要点を簡潔にまとめて回答してください。"
    )
    return prompt_default, prompt_nsfw, prompt_search

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --------------------------------------------------
# 🎨 ComfyUI 画像生成 & 受信処理
# --------------------------------------------------
def get_comfy_image(filename, subfolder, folder_type):
    """ComfyUIサーバーから生成された画像バイナリデータを取得する"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    url = f"http://{COMFYUI_SERVER}/view?{url_values}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return response.read()

async def generate_and_get_images(positive_prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024):
    """ComfyUIに生成リクエストを送り、完成した画像を取得して返す（ポーリング方式）"""
    client_id = f"telegram_{int(datetime.now().timestamp())}"
    
    if not os.path.exists(COMFYUI_WORKFLOW_PATH):
        logging.error(f"ComfyUIワークフローファイルが見つかりません: {COMFYUI_WORKFLOW_PATH}")
        return None
        
    with open(COMFYUI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow_data = json.load(f)
        
    # ノード "4" (ポジティブプロンプト)
    if "4" in workflow_data and "inputs" in workflow_data["4"]:
        workflow_data["4"]["inputs"]["text"] = positive_prompt

    # ノード "5" (ネガティブプロンプト)
    if "5" in workflow_data and "inputs" in workflow_data["5"]:
        workflow_data["5"]["inputs"]["text"] = negative_prompt

    # ノード "6" (Empty Latent Image) のサイズを変更
    if "6" in workflow_data and "inputs" in workflow_data["6"]:
        workflow_data["6"]["inputs"]["width"] = width
        workflow_data["6"]["inputs"]["height"] = height

    p = {"prompt": workflow_data, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(COMFYUI_URL, data=data, headers={'Content-Type': 'application/json'})
    
    # 1. 画像生成リクエスト送信
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            prompt_id = res.get('prompt_id')
    except Exception as e:
        logging.error(f"ComfyUIリクエスト送信エラー: {e}")
        return None

    if not prompt_id:
        return None

    # 2. 履歴APIをポーリングして生成完了を監視（タイムアウト: 180秒）
    start_time = datetime.now()
    images_binary = []

    while (datetime.now() - start_time).total_seconds() < 180:
        try:
            history_url = f"http://{COMFYUI_SERVER}/history/{prompt_id}"
            def fetch_history():
                with urllib.request.urlopen(history_url) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            
            history = await asyncio.to_thread(fetch_history)
            
            if prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                if outputs:
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output:
                            for image_info in node_output['images']:
                                img_data = await asyncio.to_thread(
                                    get_comfy_image, 
                                    image_info['filename'], 
                                    image_info['subfolder'], 
                                    image_info['type']
                                )
                                images_binary.append(img_data)
                    break
        except Exception:
            pass
            
        await asyncio.sleep(2)

    return images_binary if images_binary else None

# --------------------------------------------------
# 🔍 検索・Gemini関連
# --------------------------------------------------
async def search_duckduckgo_async(query: str, max_results: int = 5):
    return await asyncio.to_thread(search_duckduckgo_with_urls, query, max_results)

def search_duckduckgo_with_urls(query: str, max_results: int = 5) -> list:
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        data = {"q": query}
        response = requests.post(url, headers=headers, data=data, timeout=10)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.text, "html.parser")
        result_elements = soup.select(".result")
        
        results = []
        for element in result_elements:
            if len(results) >= max_results:
                break
            title_tag = element.select_one(".result__title")
            snippet_tag = element.select_one(".result__snippet")
            link_tag = element.select_one(".result__url")
            
            if title_tag and link_tag:
                title = title_tag.get_text(strip=True)
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                raw_href = link_tag.get("href", "")
                target_url = raw_href
                if "uddg=" in raw_href:
                    from urllib.parse import parse_qs, urlparse
                    parsed_url = urlparse(raw_href)
                    query_params = parse_qs(parsed_url.query)
                    if "uddg" in query_params:
                        target_url = query_params["uddg"][0]
                
                results.append({"title": title, "snippet": snippet, "url": target_url})
        return results
    except Exception as e:
        logging.error(f"DuckDuckGo検索エラー: {e}")
        return []

def scrape_page_text(url: str, max_chars: int = 3000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception as e:
        logging.error(f"スクレイピングエラー ({url}): {e}")
        return ""

async def is_search_needed_async(user_text: str) -> bool:
    try:
        prompt = (
            f"現在の日付は {datetime.now().strftime('%Y年%m月%d日')} です。\n"
            "以下の入力に対してWeb検索が必要か判断してください。「YES」または「NO」のみで回答。\n"
            f"入力: {user_text}"
        )
        response = await router_model.generate_content_async(prompt)
        return "YES" in response.text.strip().upper()
    except Exception as e:
        return False

async def is_nsfw_writing_request_async(user_text: str, replied_text: str = "") -> bool:
    try:
        context_hint = f"\n直近の文脈: {replied_text}" if replied_text else ""
        prompt = (
            "以下の入力が官能小説やアダルト文章の創作に該当するか判断してください。「YES」または「NO」のみで回答。\n"
            f"入力: {user_text}{context_hint}"
        )
        response = await router_model.generate_content_async(prompt)
        return "YES" in response.text.strip().upper()
    except Exception as e:
        return True

async def select_best_url_and_gather_data_async(user_text: str, search_results: list) -> str:
    snippets_text = "【検索結果概要】\n"
    for i, res in enumerate(search_results):
        snippets_text += f"{i+1}. {res['title']}\n   {res['url']}\n   {res['snippet']}\n"
    try:
        prompt = f"ユーザー要望: 「{user_text}」\n最も詳しい情報が載っているURLの番号(1〜5)を1つだけ選んでください。"
        response = await router_model.generate_content_async(prompt + "\n\n" + snippets_text)
        import re
        match = re.search(r'\d+', response.text.strip())
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(search_results):
                page_text = await asyncio.to_thread(scrape_page_text, search_results[idx]['url'])
                if page_text and len(page_text.strip()) > 100:
                    return f"【本文データ】\n{page_text}\n\n{snippets_text}"
    except Exception as e:
        logging.error(f"URL選定エラー: {e}")
    return snippets_text

# --------------------------------------------------
# 💬 メッセージ処理メイン
# --------------------------------------------------
async def process_single_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return

        chat_id = update.effective_chat.id
        raw_text = update.message.text
        bot_username = context.bot.username.lower() if context.bot.username else ""

        clean_text = raw_text
        if bot_username:
            clean_text = clean_text.replace(f"@{context.bot.username}", "").replace(f"@{bot_username}", "")
        clean_text = clean_text.strip()

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # 🎨 "img:" または "生成:" で始まっている場合の画像生成処理
        if clean_text.startswith("img:") or clean_text.startswith("生成:"):
            body = clean_text.replace("img:", "", 1).replace("生成:", "", 1).strip()
            
            if not body:
                usage_text = (
                    "🎨 **ComfyUI 画像生成の使い方**\n\n"
                    "• スクエア (1024x1024):\n  `img: 1girl, cat ears`\n\n"
                    "• ネガティブ指定:\n  `img: 1girl / blurry`\n\n"
                    "• サイズ指定（3番目に v, h, s を指定）:\n"
                    "  `img: 1girl / / v` （縦長：896x1152）\n"
                    "  `img: 1girl / / h` （横長：1152x896）\n"
                    "  `img: 1girl / blurry / s` （スクエア：1024x1024）"
                )
                await update.message.reply_text(usage_text, parse_mode="Markdown", reply_to_message_id=update.message.message_id)
                return

            positive_prompt = ""
            negative_prompt = ""
            size_mode = "s"  # デフォルトはスクエア

            # スラッシュで分割 (最大3パート: [ポジティブ, ネガティブ, サイズ])
            parts = [p.strip() for p in body.split("/")]
            
            if len(parts) >= 1:
                positive_prompt = parts[0]
            if len(parts) >= 2:
                negative_prompt = parts[1]
            if len(parts) >= 3 and parts[2]:
                size_mode = parts[2].lower()

            # サイズモードに応じた解像度の割り当て（指定なし・未入力の場合はスクエア）
            width, height = 1024, 1024
            size_desc = "スクエア (1024x1024)"
            
            if size_mode == "v":
                width, height = 896, 1152
                size_desc = "縦長 (896x1152)"
            elif size_mode == "h":
                width, height = 1152, 896
                size_desc = "横長 (1152x896)"
            elif size_mode == "s":
                width, height = 1024, 1024
                size_desc = "スクエア (1024x1024)"

            status_msg = await update.message.reply_text(
                f"🎨 画像を生成中です...少々お待ちください。\n"
                f"👍 ポジティブ: {positive_prompt}\n"
                f"👎 ネガティブ: {negative_prompt if negative_prompt else '(なし)'}\n"
                f"📐 サイズ: {size_desc}",
                reply_to_message_id=update.message.message_id
            )

            # 画像生成とダウンロードを実行（サイズを渡す）
            images = await generate_and_get_images(positive_prompt, negative_prompt, width, height)

            if images:
                await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
                for img_bytes in images:
                    bio = io.BytesIO(img_bytes)
                    bio.name = 'image.png'  # Telegramが画像として正しく認識するためのファイル名指定
                    
                    # 全ての画像を強制的にスポイラー付き（has_spoiler=True）で送信
                    await update.message.reply_photo(
                        photo=bio,
                        caption=f"✨ 生成完了! ({size_desc})\nPrompt: {positive_prompt}",
                        reply_to_message_id=update.message.message_id,
                        has_spoiler=True,
                        read_timeout=120,
                        write_timeout=120,
                        connect_timeout=60
                    )
            else:
                await update.message.reply_text("❌ 画像の生成または取得に失敗しました。ComfyUIが起動しているか確認してください。", reply_to_message_id=update.message.message_id)
            return

        # ―― 通常のチャット処理 ――
        history = chat_histories[chat_id]
        combined_context_text = " ".join([item["content"] for item in history]) + " " + clean_text

        is_nsfw = await is_nsfw_writing_request_async(clean_text, combined_context_text)
        needs_search = await is_search_needed_async(clean_text) if not is_nsfw else False
        
        search_context = ""
        if needs_search:
            search_results = await search_duckduckgo_async(clean_text, max_results=5)
            if search_results:
                search_context = await select_best_url_and_gather_data_async(clean_text, search_results)

        prompt_default, prompt_nsfw, prompt_search = get_system_prompts()
        system_instruction = prompt_nsfw if is_nsfw else (prompt_search if needs_search else prompt_default)

        final_user_content = f"{search_context}\n\n要望: {clean_text}" if search_context else clean_text

        messages_payload = [{"role": "system", "content": system_instruction}]
        for hist in history:
            messages_payload.append({"role": hist["role"], "content": hist["content"]})
        messages_payload.append({"role": "user", "content": final_user_content})

        payload = {
            "model": MODEL_NAME,
            "messages": messages_payload,
            "temperature": 0.7 if is_nsfw else 0.1
        }

        def call_ollama():
            return requests.post(OLLAMA_API_URL, json=payload, timeout=180)

        response = await asyncio.to_thread(call_ollama)
        if response.status_code == 200:
            ai_reply = response.json()['choices'][0]['message']['content']
            history.append({"role": "user", "content": clean_text})
            history.append({"role": "assistant", "content": ai_reply})
            await update.message.reply_text(ai_reply, reply_to_message_id=update.message.message_id)
        else:
            await update.message.reply_text(f"APIエラー: Status {response.status_code}")
            
    except Exception as e:
        await update.message.reply_text(f"エラーが発生しました: {str(e)}")

async def queue_worker():
    global is_busy
    while True:
        try:
            update, context = await message_queue.get()
            is_busy = True
            try:
                await process_single_message(update, context)
            except Exception as e:
                logging.error(f"ワーカー処理エラー: {e}")
            finally:
                is_busy = False
                message_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            is_busy = False

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_busy
    if not update.message or not update.message.text:
        return

    raw_text = update.message.text
    chat_type = update.effective_chat.type
    bot_username = context.bot.username.lower() if context.bot.username else ""

    if chat_type in ['group', 'supergroup']:
        is_mentioned = bot_username and (f"@{bot_username}" in raw_text.lower())
        is_reply_to_bot = (
            update.message.reply_to_message and 
            update.message.reply_to_message.from_user and 
            update.message.reply_to_message.from_user.is_bot and 
            context.bot.id == update.message.reply_to_message.from_user.id
        )
        if not is_mentioned and not is_reply_to_bot:
            return

    if is_busy:
        await update.message.reply_text("現在、別の処理を実行中です。完了まで少しお待ちください。", reply_to_message_id=update.message.message_id)
        return

    await message_queue.put((update, context))

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    async def post_init(application):
        global message_queue
        message_queue = asyncio.Queue()
        application.create_task(queue_worker())
        print("Telegram Bot (サイズ補完・アナウンス対応版) 起動完了!")

    app.post_init = post_init
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()
