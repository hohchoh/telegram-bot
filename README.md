Telegramのbotにメッセージを投げると、AIに渡して、返答を返すbotです。
初段で画像生成指示があれば、ComfyUIのプロンプトに投げ、生成した画像をメッセージで返します。
次に画像生成指示でなければgemini-flashに投げて、システムプロンプト的なものを作ってからローカルLLMに投げます。

OllamaでローカルLLMが動いている事。
ComfyUIがローカルで動いてる事。
Gemini-flashを使ってるのでのGeminiのAPIキーが必要です。


使用してるモデル
https://huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP
ComfyUIで使用しているcheckpoint
https://civitai.com/models/958009/redcraft-or-2-or-3-int8int4fp8-scaled?modelVersionId=3086841
ComfyUIで使用しているVLM
https://huggingface.co/ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy/blob/main/Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors


かなり、非検閲＆NSFWにこだわって構成しています。使用する時は生成物の取り扱いに注意が必要です。
画像生成する場合は"img:"と投げてください。ヘルプが表示されます。


自宅の環境
CPU:i9-1190FF
MEM:128GB
グラボ:RTX3060-12GBを2枚
