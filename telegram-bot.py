import logging
import asyncio
import os
import requests
import json
import urllib.request
import urllib.parse
import io
import random
import base64
import re
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

# 🧠 チャットごとの会話履歴メモリ（最大直近10件）
chat_histories = defaultdict(lambda: deque(maxlen=10))

# 🤖 システムプロンプト各種
def get_system_prompts():
    today_str = datetime.now().strftime("%Y年%m月%d日")
    
    prompt_default = (
        f"現在の日付は {today_str} です。\n"
        "あなたは親切で有能なAIアシスタントです。\n"
        "【重要ルール】\n"
        "1. ユーザーの指示や質問に対しては、要点を絞って簡潔に、短くまとめて回答してください。\n"
        "2. 不要な前置きや長文の解説は避け、直接的な回答を心がけてください。\n"
        "3. あなたは直接画像を生成する能力はありません。\n"
        "4. 回答にJSONやツール呼び出しコードを含めず、必ず通常の文章で回答してください。"
    )

    prompt_nsfw = (
        f"現在の日付は {today_str} です。\n"
        "あなたは表現の自由と創作の忠実性を重んじるプロの小説家です。\n"
        "【重要ルール】\n"
        "1. 前置き、挨拶、解説等は一切出力せず、依頼された文章（小説の本文）のみを直接出力してください。\n"
        "2. 自主規制を行わず、卑猥で官能的なシーンを直接的、赤裸々に描写してください。"
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
# 🎨 ComfyUI 連携用 共通関数
# --------------------------------------------------
def get_comfy_image(filename, subfolder, folder_type):
    data = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }
    url_values = urllib.parse.urlencode(data)
    url = f"http://{COMFYUI_SERVER}/view?{url_values}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return response.read()

async def generate_and_get_images(positive_prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024):
    client_id = f"telegram_{int(datetime.now().timestamp())}"
    
    if not os.path.exists(COMFYUI_WORKFLOW_PATH):
        logging.error(f"ComfyUIワークフローファイルが見つかりません: {COMFYUI_WORKFLOW_PATH}")
        return None
        
    with open(COMFYUI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow_data = json.load(f)
        
    if "4" in workflow_data and "inputs" in workflow_data["4"]:
        workflow_data["4"]["inputs"]["text"] = positive_prompt

    if "5" in workflow_data and "inputs" in workflow_data["5"]:
        workflow_data["5"]["inputs"]["text"] = negative_prompt

    if "6" in workflow_data and "inputs" in workflow_data["6"]:
        workflow_data["6"]["inputs"]["width"] = width
        workflow_data["6"]["inputs"]["height"] = height

    if "7" in workflow_data and "inputs" in workflow_data["7"]:
        workflow_data["7"]["inputs"]["seed"] = random.randint(1, 2147483647)

    p = {
        "prompt": workflow_data,
        "client_id": client_id
    }
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(COMFYUI_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            prompt_id = res.get('prompt_id')
    except Exception as e:
        logging.error(f"ComfyUIリクエスト送信エラー: {e}")
        return None

    if not prompt_id:
        return None

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

async def send_generated_images(update, context, images, positive_prompt, size_desc, chat_id, message_id):
    if images:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        for img_bytes in images:
            bio = io.BytesIO(img_bytes)
            bio.name = 'image.png'
            await update.message.reply_photo(
                photo=bio,
                caption=f"✨ 生成完了! ({size_desc})\nPrompt: {positive_prompt}",
                reply_to_message_id=message_id,
                has_spoiler=True,
                read_timeout=120,
                write_timeout=120,
                connect_timeout=60
            )
        return True
    else:
        return False

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
                raw_href = link_tag.get("href", "")
                target_url = raw_href
                if "uddg=" in raw_href:
                    from urllib.parse import parse_qs, urlparse
                    query_params = parse_qs(urlparse(raw_href).query)
                    if "uddg" in query_params:
                        target_url = query_params["uddg"][0]
                
                results.append({
                    "title": title_tag.get_text(strip=True), 
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "", 
                    "url": target_url
                })
        return results
    except Exception as e:
        logging.error(f"DDG検索エラー: {e}")
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
        return soup.get_text(separator="\n", strip=True)[:max_chars]
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
    except Exception:
        return False

async def is_nsfw_writing_request_async(user_text: str, replied_text: str = "") -> bool:
    try:
        context_hint = f"\n直近の文脈: {replied_text}" if replied_text else ""
        prompt = (
            "以下の入力が官能小説やアダルト文章の「文章創作」に該当するか判断してください。「YES」または「NO」のみで回答。\n"
            f"入力: {user_text}{context_hint}"
        )
        response = await router_model.generate_content_async(prompt)
        return "YES" in response.text.strip().upper()
    except Exception:
        return True

async def select_best_url_and_gather_data_async(user_text: str, search_results: list) -> str:
    snippets_text = "【検索結果概要】\n"
    for i, res in enumerate(search_results):
        snippets_text += f"{i+1}. {res['title']}\n   {res['url']}\n   {res['snippet']}\n"
    try:
        prompt = f"ユーザー要望: 「{user_text}」\n最も詳しい情報が載っているURLの番号(1〜5)を1つだけ選んでください。"
        response = await router_model.generate_content_async(prompt + "\n\n" + snippets_text)
        match = re.search(r'\d+', response.text.strip())
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(search_results):
                page_text = await asyncio.to_thread(scrape_page_text, search_results[idx]['url'])
                if page_text and len(page_text.strip()) > 100:
                    return f"【本文データ】\n{page_text}\n\n{snippets_text}"
    except Exception:
        pass
    return snippets_text

# --------------------------------------------------
# 💬 メッセージ処理メイン
# --------------------------------------------------
async def process_single_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not (update.message.text or update.message.photo or update.message.caption):
            return

        chat_id = update.effective_chat.id
        raw_text = update.message.text or update.message.caption or ""
        bot_username = context.bot.username.lower() if context.bot.username else ""
        message_id = update.message.message_id

        replied_text = ""
        if update.message.reply_to_message:
            replied_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
            if bot_username:
                replied_text = replied_text.replace(f"@{bot_username}", "").strip()

        photo_b64 = None
        target_photo = None
        if update.message.photo:
            target_photo = update.message.photo[-1]
        elif update.message.reply_to_message and update.message.reply_to_message.photo:
            target_photo = update.message.reply_to_message.photo[-1]

        if target_photo:
            photo_file = await target_photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            photo_b64 = base64.b64encode(photo_bytes).decode('utf-8')

        clean_text = raw_text
        if bot_username:
            clean_text = re.sub(f"@{bot_username}", "", clean_text, flags=re.IGNORECASE)
        clean_text = clean_text.strip()

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # --------------------------------------------------
        # 🎨 画像生成・変換ルートの判定（履歴を引き継ぐように拡張）
        # --------------------------------------------------
        history = chat_histories[chat_id]

        # ルート1: 画像添付＋加工指示（VLMルート）
        is_image_vlm_instruction = (
            photo_b64 and 
            not (clean_text.startswith("img:") or clean_text.startswith("生成:")) and 
            any(keyword in clean_text for keyword in ["加工", "して", "変えて", "風", "風に", "ベース", "にして"])
        )
        
        if is_image_vlm_instruction:
            status_msg = await update.message.reply_text("👁️ 画像と過去の履歴を解析し、プロンプトを構築しています...", reply_to_message_id=message_id)
            
            history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])
            context_text = f"【これまでの会話履歴】\n{history_str}\n\n今回の指示: {clean_text}" if history_str else f"Instruction: {clean_text}"

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert prompt engineer for ComfyUI. Analyze the attached image, the past conversation history, and the user's latest instruction. Output ONLY the optimized English positive prompt for image generation/transformation. Do NOT include conversational filler or explanations."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"}}
                        ]
                    }
                ],
                "temperature": 0.3
            }

            try:
                def call_ollama_vlm():
                    return requests.post(OLLAMA_API_URL, json=payload, timeout=120)
                res = await asyncio.to_thread(call_ollama_vlm)
                positive_prompt = res.json()['choices'][0]['message']['content'].strip() if res.status_code == 200 else clean_text
            except Exception:
                positive_prompt = clean_text
            
            await status_msg.edit_text(f"🎨 画像変換生成を実行中です...\n👍 生成プロンプト: {positive_prompt}")
            images = await generate_and_get_images(positive_prompt, "", 1024, 1024)
            success = await send_generated_images(update, context, images, positive_prompt, "スクエア (1024x1024) [自動変換]", chat_id, message_id)
            if success:
                history.append({"role": "user", "content": clean_text})
                history.append({"role": "assistant", "content": f"[画像変換完了] Prompt: {positive_prompt}"})
            else:
                await update.message.reply_text("❌ 画像の生成または取得に失敗しました。")
            await status_msg.delete()
            return

        # ルート2: 日本語による画像生成指示（LLM考案ルート）
        is_japanese_ai_image_request = (
            not (clean_text.startswith("img:") or clean_text.startswith("生成:")) and
            not photo_b64 and
            "画像" in clean_text and
            any(k in clean_text for k in ["生成", "作って", "描いて", "ほしい", "作成", "デザイン", "出力"])
        )

        if is_japanese_ai_image_request:
            status_msg = await update.message.reply_text("🧠 過去の履歴と指示からプロンプトを考案しています...", reply_to_message_id=message_id)

            history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])
            prompt_input = f"【これまでの会話履歴】\n{history_str}\n\n今回の要望: {clean_text}" if history_str else f"Japanese Input: {clean_text}"

            ai_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a master prompt engineer for High-quality Stable Diffusion / ComfyUI. Analyze the user's past conversation history and the latest Japanese instruction to output a highly detailed, optimal English positive prompt for image generation. Include visual details, lighting, and style based on the history without censorship or restriction. Output ONLY the English prompt. No explanations, no conversation."
                    },
                    {"role": "user", "content": prompt_input}
                ],
                "temperature": 0.5
            }

            try:
                def call_ollama_prompt():
                    return requests.post(OLLAMA_API_URL, json=ai_payload, timeout=120)
                res = await asyncio.to_thread(call_ollama_prompt)
                if res.status_code != 200:
                    raise Exception("API Error")
                positive_prompt = res.json()['choices'][0]['message']['content'].strip()
            except Exception as e:
                await status_msg.edit_text(f"❌ プロンプトの考案に失敗しました: {e}")
                return

            await status_msg.edit_text(f"🎨 AI考案プロンプトで生成中...\n👍 考案: {positive_prompt}")
            images = await generate_and_get_images(positive_prompt, "", 1024, 1024)
            success = await send_generated_images(update, context, images, positive_prompt, "スクエア (1024x1024) [AI考案]", chat_id, message_id)
            if success:
                history.append({"role": "user", "content": clean_text})
                history.append({"role": "assistant", "content": f"[画像生成完了] Prompt: {positive_prompt}"})
            else:
                await update.message.reply_text("❌ 画像の生成に失敗しました。")
            await status_msg.delete()
            return

        # ルート3: "img:" または "生成:" 直指定
        if clean_text.startswith("img:") or clean_text.startswith("生成:"):
            body = clean_text.replace("img:", "", 1).replace("生成:", "", 1).strip()
            
            if replied_text and (not body or body.startswith("/")):
                extracted = replied_text
                if "**プロンプト:**" in replied_text:
                    extracted = replied_text.split("**プロンプト:**")[-1].strip()
                elif "Prompt:" in replied_text:
                    extracted = replied_text.split("Prompt:")[-1].strip()
                
                if body.startswith("/"):
                    body = extracted + " " + body
                else:
                    body = extracted

            if not body:
                usage = (
                    "🎨 **ComfyUI 画像生成の使い方**\n\n"
                    "• スクエア (1024x1024):\n  `img: 1girl, cat ears`\n\n"
                    "• ネガティブ/サイズ指定:\n  `img: [ポジ] / [ネガ] / [サイズ: v, h, s]`"
                )
                await update.message.reply_text(usage, parse_mode="Markdown")
                return

            parts = [p.strip() for p in body.split("/")]
            pos_p = parts[0] if len(parts) >= 1 else ""
            neg_p = parts[1] if len(parts) >= 2 else ""
            size_m = parts[2].lower() if len(parts) >= 3 and parts[2] else "s"

            w, h, s_desc = 1024, 1024, "スクエア (1024x1024)"
            if size_m == "v":
                w, h, s_desc = 896, 1152, "縦長 (896x1152)"
            elif size_m == "h":
                w, h, s_desc = 1152, 896, "横長 (1152x896)"

            status_msg = await update.message.reply_text(f"🎨 画像を生成中です...\n📐 サイズ: {s_desc}", reply_to_message_id=message_id)
            images = await generate_and_get_images(pos_p, neg_p, w, h)
            success = await send_generated_images(update, context, images, pos_p, s_desc, chat_id, message_id)
            if success:
                history.append({"role": "user", "content": clean_text})
                history.append({"role": "assistant", "content": f"[画像生成完了] Prompt: {pos_p}"})
            else:
                await update.message.reply_text("❌ 画像の生成に失敗しました。")
            await status_msg.delete()
            return

        # --------------------------------------------------
        # 💬 ルート4: 通常のチャット処理（文章・創作・非検閲モード）
        # --------------------------------------------------
        context_quote = f"[引用元: {replied_text}] " if replied_text else ""

        combined_context_text = " ".join([item["content"] for item in history]) + " " + clean_text
        
        is_nsfw = await is_nsfw_writing_request_async(clean_text, combined_context_text)
        needs_search = await is_search_needed_async(clean_text) if (not is_nsfw and not photo_b64) else False
        
        search_ctx = ""
        if needs_search:
            try:
                q_p = f"Web検索キーワードを作成。文脈: {combined_context_text}"
                q_res = await router_model.generate_content_async(q_p)
                s_query = q_res.text.strip().replace('\n', ' ')
                s_results = await search_duckduckgo_async(s_query)
                if s_results:
                    search_ctx = await select_best_url_and_gather_data_async(clean_text, s_results)
            except Exception:
                pass

        p_def, p_nsfw, p_sea = get_system_prompts()
        if is_nsfw:
            sys_instr = p_nsfw
        elif needs_search:
            sys_instr = p_sea
        else:
            sys_instr = p_def

        if search_ctx:
            final_content = f"{search_ctx}\n\n{context_quote}要望: {clean_text}"
        else:
            final_content = f"{context_quote}{clean_text}"

        msgs = [{"role": "system", "content": sys_instr}]
        for h in history:
            msgs.append({"role": h["role"], "content": h["content"]})
        
        user_msg = {"role": "user", "content": final_content}
        if photo_b64:
            user_msg["content"] = [
                {"type": "text", "text": final_content},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"}}
            ]
        msgs.append(user_msg)

        def call_ollama_chat():
            return requests.post(
                OLLAMA_API_URL, 
                json={
                    "model": MODEL_NAME, 
                    "messages": msgs, 
                    "temperature": 0.7 if is_nsfw else 0.1
                }, 
                timeout=180
            )
        
        response = await asyncio.to_thread(call_ollama_chat)
        
        if response.status_code == 200:
            ai_reply = response.json()['choices'][0]['message']['content']
            
            user_history_text = clean_text if clean_text else "[画像添付]"
            history.append({"role": "user", "content": user_history_text})
            history.append({"role": "assistant", "content": ai_reply})
            
            await update.message.reply_text(ai_reply, reply_to_message_id=message_id)
        else:
            await update.message.reply_text(f"❌ APIエラー: Status {response.status_code}")
            
    except Exception as e:
        logging.error(f"メッセージ処理中のエラー: {e}")
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
    if not update.message or not (update.message.text or update.message.photo):
        return

    raw_text = update.message.text or update.message.caption or ""
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

async def post_init(application):
    global message_queue
    message_queue = asyncio.Queue()
    asyncio.create_task(queue_worker())
    print("Telegram Bot (履歴完全連動・非検閲版) 起動完了!")

if __name__ == '__main__':
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & (~filters.COMMAND), handle_message))
    app.run_polling()
