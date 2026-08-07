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
import html
import time
from datetime import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from collections import defaultdict, deque
from dotenv import load_dotenv
from filelock import FileLock, Timeout

# --------------------------------------------------
# 🔑 設定情報 (.env から読み込み)
# --------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "hf.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP:Q4_K_M"

# 🔍 SearXNG 連携設定
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888/search")

# 🎨 ComfyUI 連携設定
COMFYUI_SERVER = "127.0.0.1:8188"
COMFYUI_URL = f"http://{COMFYUI_SERVER}/prompt"
COMFYUI_WORKFLOW_PATH = "Krea2.json"

# 🔒 排他制御用ファイルロック
GPU_LOCK_FILE = "gpu_process.lock"
gpu_lock = FileLock(GPU_LOCK_FILE)

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

# 📝 ログ出力の設定
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("BotLogger")

# 🌤 気象庁マスターデータキャッシュ用グローバル変数
JMA_AREA_MASTER = {}

# 🤖 システムプロンプト各種
def get_system_prompts():
    today_str = datetime.now().strftime("%Y年%m月%d日")
    
    prompt_default = (
        f"現在の日付は {today_str} です。\n"
        "あなたは親切で有能なAIアシスタントです。\n"
        "【重要ルール】\n"
        "1. ユーザーの指示や質問に対しては、要点を絞って簡潔に、短くまとめて回答してください。\n"
        "2. 不要な前置きや長文の解説は避け、直接的な回答を心がけてください。\n"
        "3. 画像が添付されている場合は、その画像の内容をしっかりと観察・認識して回答してください。\n"
        "4. 回答にJSONやツール呼び出しコードを含めず、必ず通常の文章で回答してください。\n"
        "5. AA（アスキーアート）やプログラムコードなどの作成・出力指示がある場合は、上記「簡潔にまとめる」という制約を無視し、指示通りの形式で直接出力してください。"
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
        "1. 提供された【気象データ】および【検索情報】がある場合は、それらを根拠にして要点を簡潔にまとめて回答してください。\n"
        "2. 天気に関する質問に答える際は、ユーザーが尋ねた対象スポットと所在地（判明している場合）を明記し、指定された時間軸（今日、明日、明後日、複数日など）に合わせて自然な文章でまとめて回答してください。"
    )
    return prompt_default, prompt_nsfw, prompt_search

# --------------------------------------------------
# 🛠 長文メッセージ分割送信用ヘルパー
# --------------------------------------------------
async def send_split_message(update: Update, text: str, parse_mode: str = "HTML", reply_to_message_id: int = None):
    """Telegramの4096文字制限に対応し、安全に分割送信する"""
    max_length = 4000
    if len(text) <= max_length:
        try:
            await update.message.reply_text(text, parse_mode=parse_mode, reply_to_message_id=reply_to_message_id)
        except Exception as parse_err:
            logger.warning(f"[HTML送信エラー] パースに失敗したためプレーンテキストで送信します: {parse_err}")
            await update.message.reply_text(text, reply_to_message_id=reply_to_message_id)
        return

    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode, reply_to_message_id=reply_to_message_id if i == 0 else None)
        except Exception:
            await update.message.reply_text(chunk, reply_to_message_id=reply_to_message_id if i == 0 else None)

# --------------------------------------------------
# ☀️ 天気情報取得 (JMA area.json + Gemini補完 + Open-Meteo + Webスクレイピング)
# --------------------------------------------------
def load_jma_area_master():
    global JMA_AREA_MASTER
    url = "https://www.jma.go.jp/bosai/common/const/area.json"
    logger.info("[JMAマスター] 気象庁公式 area.json の取得・解析を開始します...")
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            logger.error(f"[JMAマスターエラー] HTTP Status: {res.status_code}")
            return
            
        data = res.json()
        mapping = {}

        offices = data.get("offices", {})
        class10s = data.get("class10s", {})
        class20s = data.get("class20s", {})

        for code, info in offices.items():
            official_name = info.get("name", "")
            if not official_name:
                continue
            mapping[official_name] = (code, official_name)
            short_name = re.sub(r'(都|府|県)$', '', official_name)
            if len(short_name) >= 2 and short_name != official_name:
                mapping[short_name] = (code, official_name)

        for code, info in class10s.items():
            name = info.get("name", "")
            parent_office = info.get("parent", "")
            if parent_office in offices and name:
                office_name = offices[parent_office].get("name", "")
                mapping[name] = (parent_office, office_name)

        for code, info in class20s.items():
            name = info.get("name", "")
            parent10 = info.get("parent", "")
            if parent10 in class10s:
                parent_office = class10s[parent10].get("parent", "")
                if parent_office in offices and name:
                    office_name = offices[parent_office].get("name", "")
                    mapping[name] = (parent_office, office_name)
                    short_city = re.sub(r'(市|区|町|村)$', '', name)
                    if len(short_city) >= 2 and short_city not in mapping:
                        mapping[short_city] = (parent_office, office_name)

        JMA_AREA_MASTER = mapping
        logger.info(f"[JMAマスター成功] 全 {len(mapping)} 件の地域マッピングを自動構築しました。")
    except Exception as e:
        logger.error(f"[JMAマスターロード例外エラー] {e}", exc_info=True)

def find_jma_code(user_text: str):
    if not JMA_AREA_MASTER:
        load_jma_area_master()
    
    sorted_keys = sorted(JMA_AREA_MASTER.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if len(key) >= 2 and key in user_text:
            code, official_name = JMA_AREA_MASTER[key]
            logger.info(f"[JMAヒット] キーワード '{key}' -> エリアコード: {code} ({official_name})")
            return code, official_name
    return None, None

def get_jma_weather(area_code: str, location_name: str) -> str:
    logger.info(f"[気象庁API] {location_name} (コード:{area_code}) の天気データ取得を開始")
    try:
        url = f"https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json"
        res = requests.get(url, timeout=5).json()
        
        time_series = res[0]["timeSeries"][0]
        areas = time_series["areas"][0]
        
        today_weather = areas["weathers"][0]
        today_weather_clean = " ".join(today_weather.split())
        today_str = datetime.now().strftime("%Y年%m月%d日")
        
        info = f"【気象庁 公式広域データ ({location_name} - {today_str})】\n"
        info += f"本日の天気: {today_weather_clean}\n"
        
        try:
            temp_series = res[0]["timeSeries"][2]
            temps = temp_series["areas"][0]["temps"]
            if len(temps) >= 2:
                info += f"予想気温: {temps[0]}℃ 〜 {temps[1]}℃\n"
        except Exception:
            pass
            
        logger.info(f"[気象庁API] 取得成功: {today_weather_clean}")
        return info
    except Exception as e:
        logger.error(f"[気象庁API エラー] {e}")
        return ""

def get_global_weather_open_meteo(user_text: str) -> str:
    logger.info(f"[Open-Meteo] 検索クエリ: '{user_text}'")
    try:
        search_name = user_text.replace("の天気", "").replace("天気", "").replace("教えて", "").strip()
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(search_name)}&count=1&language=ja&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        results = geo_res.get("results")
        if not results:
            logger.warning(f"[Open-Meteo] 地名が見つかりませんでした: {search_name}")
            return ""
            
        target = results[0]
        lat, lon = target["latitude"], target["longitude"]
        place_name = target.get("name", search_name)
        country = target.get("country", "")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
        w_res = requests.get(weather_url, timeout=5).json()
        current = w_res.get("current_weather", {})

        temp = current.get("temperature")
        code = current.get("weathercode")

        weather_map = {
            0: "Clear sky (快晴)", 1: "Mainly clear (晴れ)", 2: "Partly cloudy (一部曇り)",
            3: "Overcast (曇り)", 45: "Foggy (霧)", 61: "Slight rain (雨)", 63: "Moderate rain (雨)",
            65: "Heavy rain (大雨)", 71: "Snow (雪)", 95: "Thunderstorm (雷雨)"
        }
        desc = weather_map.get(code, "Clear")
        today_str = datetime.now().strftime("%Y年%m月%d日")
        
        info = f"【Open-Meteo 気象データ ({place_name}, {country} - {today_str})】\n"
        info += f"現在の天気: {desc}\n"
        info += f"現在気温: {temp}℃\n"
        
        logger.info(f"[Open-Meteo] 取得成功: {place_name} ({desc}, {temp}℃)")
        return info
        
    except Exception as e:
        logger.error(f"[Open-Meteo 取得エラー] {e}")
        return ""

async def resolve_location_with_gemini_async(user_text: str) -> str:
    try:
        prompt = (
            f"ユーザーの入力: 「{user_text}」\n"
            "この入力に含まれる場所・スポット・ランドマークが位置する「日本の都道府県名および市区町村名（例: 東京都墨田区、千葉県浦安市、大阪府大阪市）」"
            "または「海外の主要都市名（例: パリ、ニューヨーク）」を特定してください。\n"
            "余計な挨拶や解説は一切含めず、地名（例: 千葉県浦安市）のみを1行で出力すること。"
        )
        response = await router_model.generate_content_async(prompt)
        resolved = response.text.strip()
        logger.info(f"[Gemini地名解析] '{user_text}' ➔ 解析結果: '{resolved}'")
        return resolved
    except Exception as e:
        logger.error(f"[Gemini地名解析エラー] {e}")
        return ""

async def get_hybrid_weather_context_async(user_text: str):
    base_weather = ""
    resolved_place = ""

    try:
        resolved_place = await resolve_location_with_gemini_async(user_text)
    except Exception as e:
        logger.warning(f"[地名解析エラー] {e}")

    code, name = None, None
    if resolved_place:
        code, name = await asyncio.to_thread(find_jma_code, resolved_place)

    if not code:
        code, name = await asyncio.to_thread(find_jma_code, user_text)
        if name and not resolved_place:
            resolved_place = name

    if code and name:
        logger.info(f"[天気処理] 気象庁マスターで '{name}' (コード:{code}) を検出。気象庁APIを呼び出します。")
        base_weather = await asyncio.to_thread(get_jma_weather, code, name)

    if not base_weather:
        logger.info("[天気処理] 気象庁コード未該当（海外・該当なし）。Open-Meteoを使用します。")
        base_weather = await asyncio.to_thread(get_global_weather_open_meteo, user_text)

    web_data = ""
    try:
        s_query = f"{user_text} ピンポイント天気"
        logger.info(f"[天気補足検索] SearXNG検索を実行: '{s_query}'")
        s_results = await search_duckduckgo_async(s_query, max_results=3)
        if s_results:
            web_data = await select_best_url_and_gather_data_async(user_text, s_results)
    except Exception as e:
        logger.warning(f"[天気補足検索エラー] {e}")

    combined_context = f"{base_weather}\n" if base_weather else ""
    if web_data:
        combined_context += f"\n【Web検索・ピンポイント補足データ】\n{web_data}\n"
    
    return combined_context, resolved_place

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

def generate_comfy_images_sync(positive_prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024):
    with gpu_lock:
        client_id = f"telegram_{int(datetime.now().timestamp())}"
        
        if not os.path.exists(COMFYUI_WORKFLOW_PATH):
            logger.error(f"ComfyUIワークフローファイルが見つかりません: {COMFYUI_WORKFLOW_PATH}")
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
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode('utf-8'))
                prompt_id = res.get('prompt_id')
        except Exception as e:
            logger.error(f"ComfyUIリクエスト送信エラー: {e}")
            return None

        if not prompt_id:
            return None

        start_time = datetime.now()
        images_binary = []

        while (datetime.now() - start_time).total_seconds() < 180:
            try:
                history_url = f"http://{COMFYUI_SERVER}/history/{prompt_id}"
                with urllib.request.urlopen(history_url, timeout=5) as resp:
                    history = json.loads(resp.read().decode('utf-8'))
                
                if prompt_id in history:
                    outputs = history[prompt_id].get('outputs', {})
                    if outputs:
                        for node_id, node_output in outputs.items():
                            if 'images' in node_output:
                                for image_info in node_output['images']:
                                    img_data = get_comfy_image(
                                        image_info['filename'], 
                                        image_info['subfolder'], 
                                        image_info['type']
                                    )
                                    images_binary.append(img_data)
                        break
            except Exception:
                pass
                
            time.sleep(2)

        return images_binary if images_binary else None

async def send_generated_images(update, context, images, positive_prompt, size_desc, chat_id, message_id):
    if images:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        
        # 1. 画像の送信（キャプションはシンプルに生成情報のみ）
        for img_bytes in images:
            bio = io.BytesIO(img_bytes)
            bio.name = 'image.png'
            await update.message.reply_photo(
                photo=bio,
                caption=f"✨ 生成完了! ({size_desc})",
                reply_to_message_id=message_id,
                has_spoiler=True,
                read_timeout=120,
                write_timeout=120,
                connect_timeout=60
            )
            
        # 2. プロンプト全文をテキストメッセージとして別送（4,096文字までOK）
        prompt_message = f"<b>👍 Prompt:</b>\n<code>{html.escape(positive_prompt)}</code>"
        await send_split_message(update, prompt_message, parse_mode="HTML", reply_to_message_id=message_id)
        
        return True
    else:
        return False

# --------------------------------------------------
# 🔍 検索・Gemini関連 (SearXNG API利用)
# --------------------------------------------------
async def search_duckduckgo_async(query: str, max_results: int = 5):
    return await asyncio.to_thread(search_searxng_with_urls, query, max_results)

def search_searxng_with_urls(query: str, max_results: int = 5) -> list:
    logger.info(f"[検索] SearXNGで検索を開始します: Query='{query}'")
    try:
        params = {
            "q": query,
            "format": "json"
        }
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"[検索エラー] SearXNGがステータスコード {response.status_code} を返しました。")
            return []
            
        data = response.json()
        raw_results = data.get("results", [])
        
        results = []
        for item in raw_results[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", "")
            })
            
        logger.info(f"[検索成功] {len(results)} 件の検索結果をSearXNGから取得しました。")
        return results
    except Exception as e:
        logger.error(f"[検索例外エラー] SearXNG検索中に例外が発生しました: {e}", exc_info=True)
        return []

def scrape_page_text(url: str, max_chars: int = 3000) -> str:
    logger.info(f"[スクレイピング] ターゲットURLの取得を開始: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"[スクレイピング失敗] URL: {url} | HTTP Status: {response.status_code}")
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text(separator="\n", strip=True)[:max_chars]
        logger.info(f"[スクレイピング成功] URL: {url} | 抽出文字数: {len(text)}文字")
        return text
    except Exception as e:
        logger.error(f"[スクレイピング例外エラー] URL: {url} | エラー: {e}")
        return ""

async def is_search_needed_async(user_text: str) -> bool:
    try:
        prompt = (
            f"現在の日付は {datetime.now().strftime('%Y年%m月%d日')} です。\n"
            "以下の入力に対してWeb検索が必要か判断してください。「YES」または「NO」のみで回答。\n"
            f"入力: {user_text}"
        )
        response = await router_model.generate_content_async(prompt)
        res_text = response.text.strip().upper()
        needed = "YES" in res_text
        logger.info(f"[検索判定] 結果: {res_text} -> 検索要否: {needed}")
        return needed
    except Exception as e:
        logger.error(f"[検索判定エラー] {e}")
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
        logger.info("[Gemini URL選定] Geminiに最適なURLの選定を依頼中...")
        response = await router_model.generate_content_async(prompt + "\n\n" + snippets_text)
        logger.info(f"[Gemini URL選定レスポンス]: {response.text.strip()}")
        
        match = re.search(r'\d+', response.text.strip())
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(search_results):
                target_url = search_results[idx]['url']
                logger.info(f"[Gemini 選択URL] {idx+1}番を選択: {target_url}")
                page_text = await asyncio.to_thread(scrape_page_text, target_url)
                if page_text and len(page_text.strip()) > 100:
                    return f"【本文データ】\n{page_text}\n\n{snippets_text}"
                else:
                    logger.warning("[スクレイピング警告] ページの本文が空または不十分です。検索スニペットのみ使用します。")
            else:
                logger.warning(f"[Gemini 選定警告] 範囲外の番号が指定されました: {idx+1}")
    except Exception as e:
        logger.error(f"[Gemini URL選定エラー] {e}", exc_info=True)
        
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

        # 画像の取得（送信された画像、またはリプライ先の画像）
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
        clean_text_lower = clean_text.lower()

        logger.info(f"--- [新メッセージ受信] User Input: '{clean_text}', Has Photo: {bool(photo_b64)} ---")
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        history = chat_histories[chat_id]

        # --------------------------------------------------
        # ルート1: 画像添付＋画像加工・変換指示（明確な改変キーワードがある場合のみ）
        # --------------------------------------------------
        edit_keywords = ["加工", "変えて", "変更", "修正", "描き直して", "風に", "ベースにして", "改変", "レタッチ", "フィルター", "ポーズ変えて", "衣装変えて"]
        is_image_vlm_instruction = (
            photo_b64 and 
            not (clean_text_lower.startswith("img:") or clean_text_lower.startswith("生成:")) and 
            any(keyword in clean_text for keyword in edit_keywords)
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
                        "content": (
                            "You are an expert prompt engineer for ComfyUI. "
                            "Analyze the attached image, the past conversation history, and the user's latest instruction. "
                            "Output ONLY the optimized English positive prompt for image generation/transformation. "
                            "Do NOT include conversational filler or explanations. "
                            "CRITICAL RULE: Do NOT use slashes ('/') anywhere in the prompt (e.g., write 'f2.8' or 'f 2.8' instead of 'f/2.8')."
                        )
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

            w, h, s_desc = 1024, 1024, "スクエア (1024x1024)"
            if any(k in clean_text for k in ["縦長", "縦", "ポートレート", "portrait"]):
                w, h, s_desc = 896, 1152, "縦長 (896x1152)"
            elif any(k in clean_text for k in ["横長", "横", "ワイド", "landscape"]):
                w, h, s_desc = 1152, 896, "横長 (1152x896)"

            await status_msg.edit_text(f"🎨 画像変換生成を実行中です（排他制御待機中）...\n📐 サイズ: {s_desc}\n👍 生成プロンプト: {positive_prompt}")
            
            images = await asyncio.to_thread(generate_comfy_images_sync, positive_prompt, "", w, h)
            success = await send_generated_images(update, context, images, positive_prompt, f"{s_desc} [自動変換]", chat_id, message_id)
            if success:
                history.append({"role": "user", "content": clean_text})
                history.append({"role": "assistant", "content": f"[画像変換完了] Prompt: {positive_prompt}"})
            else:
                await update.message.reply_text("❌ 画像の生成または取得に失敗しました。")
            await status_msg.delete()
            return

        # --------------------------------------------------
        # ルート2: 日本語による画像「新規」生成指示（画像が添付されていない場合のみ）
        # --------------------------------------------------
        image_keywords = ["画像", "絵", "イラスト", "写真", "アート", "図", "cg", "ポスター", "ビジュアル"]
        action_keywords = ["生成", "作って", "描いて", "ほしい", "作成", "デザイン", "出力", "描画", "書いて"]

        is_japanese_ai_image_request = (
            not photo_b64 and  # 画像が届いていない時のみ
            not (clean_text_lower.startswith("img:") or clean_text_lower.startswith("生成:")) and
            any(ik in clean_text_lower for ik in image_keywords) and
            any(ak in clean_text_lower for ak in action_keywords)
        )

        if is_japanese_ai_image_request:
            status_msg = await update.message.reply_text("🧠 過去の履歴と指示からプロンプトを考案しています...", reply_to_message_id=message_id)

            history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])
            
            weather_ctx = ""
            if any(w in clean_text for w in ["天気", "空", "気象", "雨", "晴れ", "曇り"]):
                weather_data, _ = await get_hybrid_weather_context_async(clean_text)
                if weather_data:
                    weather_ctx = f"\n【Live Weather Information】\n{weather_data}\n"

            prompt_input = f"【これまでの会話履歴】\n{history_str}\n{weather_ctx}\n今回の要望: {clean_text}" if history_str else f"{weather_ctx}Japanese Input: {clean_text}"

            ai_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a master prompt engineer for High-quality Stable Diffusion / ComfyUI. "
                            "Analyze the user's past conversation history, live weather data (if provided), and the latest Japanese instruction to output a highly detailed, optimal English positive prompt for image generation. "
                            "Reflect the actual weather, visual details, lighting, and style without censorship or restriction. "
                            "Output ONLY the English prompt. No explanations, no conversation. "
                            "CRITICAL RULE: Do NOT use slashes ('/') anywhere in the prompt (e.g., write 'f2.8' or 'f 2.8' instead of 'f/2.8')."
                        )
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

            w, h, size_desc = 1024, 1024, "スクエア (1024x1024)"
            if any(k in clean_text for k in ["縦長", "縦", "ポートレート", "portrait"]):
                w, h, size_desc = 896, 1152, "縦長 (896x1152)"
            elif any(k in clean_text for k in ["横長", "横", "ワイド", "landscape"]):
                w, h, size_desc = 1152, 896, "横長 (1152x896)"

            await status_msg.edit_text(f"🎨 AI考案プロンプトで生成中（排他制御待機中）...\n📐 サイズ: {size_desc}\n👍 考案: {positive_prompt}")
            
            images = await asyncio.to_thread(generate_comfy_images_sync, positive_prompt, "", w, h)
            success = await send_generated_images(update, context, images, positive_prompt, f"{size_desc} [AI考案]", chat_id, message_id)
            if success:
                history.append({"role": "user", "content": clean_text})
                history.append({"role": "assistant", "content": f"[画像生成完了] Prompt: {positive_prompt}"})
            else:
                await update.message.reply_text("❌ 画像の生成に失敗しました。")
            await status_msg.delete()
            return

        # --------------------------------------------------
        # ルート3: "img:" または "生成:" 直指定
        # --------------------------------------------------
        if clean_text_lower.startswith("img:") or clean_text_lower.startswith("生成:"):
            body = re.sub(r'^(img:|生成:)', '', clean_text, flags=re.IGNORECASE).strip()

            if replied_text and (not body or body.startswith("/")):
                extracted = replied_text
                for prefix in ["**プロンプト:**", "Prompt:", "[画像生成用プロンプト]"]:
                    if prefix in extracted:
                        extracted = extracted.split(prefix)[-1].strip()
                
                if body.startswith("/"):
                    body = extracted + " " + body
                else:
                    body = extracted

            body = re.sub(r'(?<=[a-zA-Z0-9])/(?=[a-zA-Z0-9])', ' ', body)

            if not body:
                usage = (
                    "🎨 **ComfyUI 画像生成の使い方**\n\n"
                    "• スクエア (1024x1024):\n  `img: 1girl, cat ears`\n\n"
                    "• ネガティブ指定 (スラッシュ1回):\n  `img: [ポジ] / [ネガ]`\n\n"
                    "• ネガティブ＋サイズ指定 (スラッシュ2回):\n  `img: [ポジ] / [ネガ] / [サイズ: v, h, s]`\n\n"
                    "• サイズのみ指定 (スラッシュ2回):\n  `img: [ポジ] // [サイズ: v, h, s]`"
                )
                await update.message.reply_text(usage, parse_mode="Markdown")
                return

            parts = [p.strip() for p in body.split("/")]
            pos_p = parts[0] if len(parts) >= 1 else ""
            neg_p = parts[1] if len(parts) >= 2 else ""
            size_m = parts[2].lower() if (len(parts) >= 3 and parts[2]) else "s"

            w, h, s_desc = 1024, 1024, "スクエア (1024x1024)"
            if size_m == "v":
                w, h, s_desc = 896, 1152, "縦長 (896x1152)"
            elif size_m == "h":
                w, h, s_desc = 1152, 896, "横長 (1152x896)"

            status_msg = await update.message.reply_text(f"🎨 画像を生成中です（排他制御待機中）...\n📐 サイズ: {s_desc}", reply_to_message_id=message_id)
            
            images = await asyncio.to_thread(generate_comfy_images_sync, pos_p, neg_p, w, h)
            success = await send_generated_images(update, context, images, pos_p, s_desc, chat_id, message_id)
            if success:
                history.append({"role": "user", "content": clean_text})
                history.append({"role": "assistant", "content": f"[画像生成完了] Prompt: {pos_p}"})
            else:
                await update.message.reply_text("❌ 画像の生成に失敗しました。")
            await status_msg.delete()
            return

        # --------------------------------------------------
        # ルート4: 通常のチャット処理（VLMによる画像解説・質問応答・Web検索含む）
        # --------------------------------------------------
        context_quote = f"[引用元: {replied_text}] " if replied_text else ""
        combined_context_text = " ".join([item["content"] for item in history]) + " " + clean_text
        
        is_nsfw = await is_nsfw_writing_request_async(clean_text, combined_context_text)
        needs_search = await is_search_needed_async(clean_text) if (not is_nsfw and not photo_b64) else False
        
        search_ctx = ""
        resolved_place = ""

        if needs_search:
            if any(w in clean_text for w in ["天気", "気温", "降水確率", "雨", "晴れ"]):
                logger.info("[気象情報] 天気キーワードを検出。ハイブリッド気象データの取得を開始します。")
                weather_info, resolved_place = await get_hybrid_weather_context_async(clean_text)
                if weather_info:
                    search_ctx = weather_info

            if not search_ctx:
                logger.info("[Web検索] 検索が必要と判断されました。処理を開始します。")
                try:
                    q_p = (
                        "以下の文脈から、Web検索で最も適した検索キーワード（単語の組み合わせ）だけを出力してください。\n"
                        "挨拶、解説、記号、見出し、箇条書きは一切含めず、検索ワード（例: 奈良 天気）のみを1行で出力すること。\n"
                        f"文脈: {combined_context_text}"
                    )
                    q_res = await router_model.generate_content_async(q_p)
                    
                    s_query = re.sub(r'[*#「」`\n]', ' ', q_res.text).strip()
                    s_query = " ".join(s_query.split())
                    
                    logger.info(f"[Web検索] 生成された検索クエリ: '{s_query}'")
                    
                    s_results = await search_duckduckgo_async(s_query)
                    if s_results:
                        search_ctx = await select_best_url_and_gather_data_async(clean_text, s_results)
                    else:
                        logger.warning("[Web検索失敗] SearXNGから有効な検索結果が得られませんでした。")
                except Exception as e:
                    logger.error(f"[Web検索例外エラー] {e}", exc_info=True)

        p_def, p_nsfw, p_sea = get_system_prompts()
        
        if is_nsfw:
            sys_instr = p_nsfw
            logger.info("[システムプロンプト] NSFW(創作)モード適用")
        elif search_ctx:
            sys_instr = p_sea
            logger.info("[システムプロンプト] 検索・気象データ活用(p_sea)モード適用")
        else:
            sys_instr = p_def
            logger.info("[システムプロンプト] 通常会話(p_def)モード適用")

        if search_ctx:
            location_tag = f"({resolved_place})" if resolved_place else ""
            target_spot = (
                clean_text.replace("の天気", "")
                .replace("を教えて", "")
                .replace("明日と明後日", "")
                .replace("明日", "")
                .replace("今日", "")
                .replace("明後日", "")
                .strip()
            )

            format_instruction = (
                f"\n\n【回答のフォーマットガイドライン】\n"
                f"- 回答の冒頭文は、対象スポット '{target_spot}' に続けて補足地域名 '{location_tag}' を入れた形式で記述してください。\n"
                f"- 質問に含まれる日付（今日/明日/明後日/複数日）の天気・気温・注意点などを分かりやすくまとめて回答してください。"
            )
            final_content = f"{search_ctx}{format_instruction}\n\n{context_quote}要望: {clean_text}"
        else:
            final_content = f"{context_quote}{clean_text}"

        msgs = [{"role": "system", "content": sys_instr}]
        for h in history:
            msgs.append({"role": h["role"], "content": h["content"]})
        
        # 💡 画像(photo_b64)が存在する場合は、LLM（Gemma 4 VLM）に画像とテキストを渡す
        if photo_b64:
            logger.info("[Ollama VLM] 画像データを含めてLLMへ要求を送信します（画像解析モード）。")
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_content if final_content else "この画像を詳しく解説・説明してください。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"}}
                ]
            }
        else:
            user_msg = {"role": "user", "content": final_content}

        msgs.append(user_msg)

        def call_ollama_chat():
            return requests.post(
                OLLAMA_API_URL, 
                json={
                    "model": MODEL_NAME, 
                    "messages": msgs, 
                    "options": {
                        "num_predict": 2048
                    },
                    "temperature": 0.7 if is_nsfw else 0.2
                }, 
                timeout=180
            )
        
        logger.info("[Ollama] LLMへの生成リクエストを送信中...")
        response = await asyncio.to_thread(call_ollama_chat)
        
        if response.status_code == 200:
            ai_reply = response.json()['choices'][0]['message']['content']
            logger.info(f"[Ollama 応答成功] 返答文字数: {len(ai_reply)}文字")
            
            user_history_text = clean_text if clean_text else "[画像添付]"
            history.append({"role": "user", "content": user_history_text})
            history.append({"role": "assistant", "content": ai_reply})
            
            safe_reply = html.escape(ai_reply)
            formatted_reply = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_reply)
            
            await send_split_message(update, formatted_reply, parse_mode="HTML", reply_to_message_id=message_id)
        else:
            logger.error(f"[Ollama エラー] Status: {response.status_code}")
            await update.message.reply_text(f"❌ APIエラー: Status {response.status_code}")
            
    except Exception as e:
        logger.error(f"メッセージ処理中の例外エラー: {e}", exc_info=True)
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
                logger.error(f"ワーカー処理エラー: {e}", exc_info=True)
            finally:
                is_busy = False
                message_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            is_busy = False

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    await message_queue.put((update, context))

async def post_init(application):
    global message_queue
    message_queue = asyncio.Queue()
    asyncio.create_task(queue_worker())
    
    await asyncio.to_thread(load_jma_area_master)
    
    logger.info("Telegram Bot 起動完了 (画像VLM解析・画像生成・天気・Web検索 判定分岐調整済み)")

if __name__ == '__main__':
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    
    app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & (~filters.COMMAND), handle_message))
    app.run_polling()
