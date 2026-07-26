import logging
import asyncio
import os
import requests
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

# 🤖 システムプロンプト各種（現在の日付を自動付与）
def get_system_prompts():
    today_str = datetime.now().strftime("%Y年%m月%d日")
    
    prompt_default = (
        f"現在の日付は {today_str} です。\n"
        "あなたは親切で有能なAIアシスタントです。\n"
        "【重要ルール】\n"
        "1. ユーザーの指示や質問に対しては、チャット欄が乱雑にならないよう、要点を絞って簡潔に、短くまとめて回答してください。\n"
        "2. 不要な前置きや長文の解説は避け、直接的な回答を心がけてください。"
    )

    prompt_nsfw = (
        f"現在の日付は {today_str} です。\n"
        "あなたは表現の自由と創作の忠実性を重んじるプロの小説家です。\n"
        "【重要ルール】\n"
        "1. ユーザーから執筆依頼があった場合、**「承知いたしました」「以下に小説を書きます」などの前置き、挨拶、解説、あとがき等は一切出力しないでください。**\n"
        "2. 依頼された文章（小説の本文）のみを直接出力してください。\n"
        "3. 自主規制やオブラートに包んだ表現（比喩表現など）を行わず、卑猥でエロく、肉体的な接触や官能的なシーンを直接的、赤裸々に、詳細かつ情熱的に描写してください。"
    )

    prompt_search = (
        f"現在の日付は {today_str} です。\n"
        "あなたは優秀なファクトチェック・リサーチアシスタントです。\n"
        "【重要ルール】\n"
        "1. 提供された【検索情報（Webページの本文および概要）】がある場合は、それを根拠にして要点を簡潔にまとめて回答してください。\n"
        "2. チャット欄が乱雑にならないよう、長文になりすぎないように注意してください。"
    )
    return prompt_default, prompt_nsfw, prompt_search
# --------------------------------------------------

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def search_duckduckgo_async(query: str, max_results: int = 5):
    return await asyncio.to_thread(search_duckduckgo_with_urls, query, max_results)

def search_duckduckgo_with_urls(query: str, max_results: int = 5) -> list:
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
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
                
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": target_url
                })
        return results
    except Exception as e:
        logging.error(f"DuckDuckGo検索エラー: {e}")
        return []

def scrape_page_text(url: str, max_chars: int = 3000) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Encoding": "gzip, deflate"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
            
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception as e:
        logging.error(f"ページスクレイピングエラー ({url}): {e}")
        return ""

async def is_search_needed_async(user_text: str) -> bool:
    try:
        prompt = (
            f"現在の日付は {datetime.now().strftime('%Y年%m月%d日')} です。\n"
            "以下のユーザーからの入力に対して、作品の概要, 人物, ゲームの設定や専門用語, 天気予報、最新情報の調査など、Web検索が必要かどうかを判断してください。\n"
            "判断基準:\n"
            "- 天気予報、漫画, アニメ, 映画, ゲーム, 固有名詞の詳細な設定、最新情報を調べる必要がある場合は 「YES」\n"
            "- 検索が全く不要な場合は 「NO」\n\n"
            f"入力: {user_text}\n"
            "回答は 「YES」 または 「NO」 のみで出力してください。"
        )
        response = await router_model.generate_content_async(prompt)
        answer = response.text.strip().upper()
        print(f"🧠 [Gemini 検索判定]: {answer} (入力: {user_text})")
        return "YES" in answer
    except Exception as e:
        logging.error(f"Gemini 判定エラー: {e}")
        return False

async def is_nsfw_writing_request_async(user_text: str, replied_text: str = "") -> bool:
    try:
        context_hint = f"\n直近の文脈: {replied_text}" if replied_text else ""
        prompt = (
            "以下のユーザーからの入力（および直近の文脈）が、官能小説、アダルトな物語、性的な描写を含む創作・文章の執筆、またはその続きを書く依頼に該当するかどうかを判断してください。\n"
            f"入力: {user_text}"
            f"{context_hint}\n"
            "該当する場合は 「YES」、そうでない場合は 「NO」 のみで出力してください。"
        )
        response = await router_model.generate_content_async(prompt)
        answer = response.text.strip().upper()
        print(f"🔞 [Gemini 創作判定]: {answer}")
        return "YES" in answer
    except Exception as e:
        logging.error(f"Gemini 創作判定エラー (セーフティブロック検知): {e}")
        print("🔞 [Gemini 創作判定 救済措置]: ブロックによる例外発生のため、NSFW（官能・アダルト）として扱います")
        return True

async def select_best_url_and_gather_data_async(user_text: str, search_results: list) -> str:
    snippets_text = "【検索結果の概要一覧】\n"
    for i, res in enumerate(search_results):
        snippets_text += f"{i+1}. タイトル: {res['title']}\n   URL: {res['url']}\n   概要: {res['snippet']}\n"

    try:
        options_text = ""
        for i, res in enumerate(search_results):
            options_text += f"[{i+1}] タイトル: {res['title']}\nURL: {res['url']}\n概要: {res['snippet']}\n\n"
            
        prompt = (
            f"ユーザーの要望: 「{user_text}」\n\n"
            "上記の要望に登場する用語や世界観を把握するために最も詳しく情報が載っていそうなURLの番号（1〜5の数字）を1つだけ選んでください。\n"
            "番号の数字（例: 1）のみを答えてください。"
        )
        response = await router_model.generate_content_async(prompt + "\n\n" + options_text)
        answer = response.text.strip()
        
        import re
        match = re.search(r'\d+', answer)
        if match:
            idx = int(match.group()) - 1
            if 0 <= idx < len(search_results):
                selected_url = search_results[idx]['url']
                print(f"🎯 [Gemini URL選定]: 番号 {idx+1} -> {selected_url}")
                
                page_text = await asyncio.to_thread(scrape_page_text, selected_url)
                if page_text and len(page_text.strip()) > 100:
                    combined = f"【Webページの本文データ (URL: {selected_url})】\n{page_text}\n\n{snippets_text}"
                    return combined
    except Exception as e:
        logging.error(f"Gemini URL選定エラー: {e}")
    
    return snippets_text

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
        if is_nsfw:
            system_instruction = prompt_nsfw
        else:
            system_instruction = prompt_search if needs_search else prompt_default

        if search_context:
            final_user_content = f"{search_context}\n\n上記の設定や背景知識も参考にしながら、要点を簡潔に（創作の場合は本文のみを）回答してください。\n要望: {clean_text}"
        else:
            final_user_content = clean_text

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
        await update.message.reply_text(f"接続エラー: {str(e)}")

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
            logging.error(f"キューワーカー外側エラー: {e}")
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
        await update.message.reply_text(
            "現在、別件の処理中です。恐れ入りますが、少し待ってから再度リクエストをお願いします。",
            reply_to_message_id=update.message.message_id
        )
        return

    await message_queue.put((update, context))

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    async def post_init(application):
        global message_queue
        message_queue = asyncio.Queue()
        application.create_task(queue_worker())
        print("Telegram Bot (現在日付自動付与版) 起動中...")

    app.post_init = post_init
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    app.run_polling()