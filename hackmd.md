**這裡蒐集在各地看到的有趣論文，同時包含資料集等可能會用到的東西，以及大家覺得有特點或是重要的東西、討論等等。**
> 在這裡的資料大家都可以隨意編輯，或是加入自己的內容，請不要不小心按到ctrl+A刪除，我會去找你。

# Multimodal
## 圖片理解(Image-to-text等等)
### [Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic](https://paperswithcode.com/paper/shikra-unleashing-multimodal-llm-s)[2023年6月27日]
![](https://raw.githubusercontent.com/shikras/shikra/main/assets/teaser.jpg =50%x)
#### 摘要
在人類對話中，個人可以在向他人講話時指出場景中的相關區域。反過來，如果有必要，對方也可以通過提及特定區域來做出回應。在當前的多模態大語言模型（MLLM）中，對話中的這種自然參考能力仍然不存在。為了填補這一空白，本文提出了一種名為 Shikra 的 MLLM，它可以處理自然語言的空間坐標輸入和輸出。其架構由視覺編碼器、對齊層和 LLM 組成。它的設計簡單明了，不需要額外的詞彙、位置編碼器、前/後檢測模塊或外部插件模型。所有輸入和輸出都是自然語言形式。參考對話是各種視覺語言（VL）任務的**超集**。Shikra 可以自然地處理與位置相關的任務，如 REC 和 PointQA，以及傳統的 VL 任務，如圖像字幕和 VQA。實驗結果展示了 Shikra 的良好性能。此外，它還支持許多令人興奮的應用，例如在思想鏈中提供提到的對象的坐標以及比較用戶指向的區域的相似性。

---------


### [Language Is Not All You Need: Aligning Perception with Language Models](https://paperswithcode.com/paper/language-is-not-all-you-need-aligning)[2023年2月27日]
![](https://hackmd.io/_uploads/r1LoYomK2.png)
[microsoft/unilm](https://github.com/microsoft/unilm)
#### 摘要
**有很多不同任務的資料集整理。**
語言、多模態感知、行動和世界建模的大融合是邁向通用人工智能的關鍵一步。在這項工作中，我們介紹了 Kosmos-1，一種多模態大型語言模型 (MLLM)，它可以感知一般模態、在上下文中學習（即少樣本）並遵循指令（即零樣本）。具體來說，我們在網絡規模的多模態語料庫上從頭開始訓練 Kosmos-1，包括任意交錯的文本和圖像、圖像標題對和文本數據。我們在沒有任何梯度更新或微調的情況下，在廣泛的任務中評估各種設置，包括零樣本、少樣本和多模式思維鏈提示。實驗結果表明，Kosmos-1 在以下方面取得了令人印象深刻的性能：（i）語言理解、生成，甚至無需 OCR 的 NLP（直接輸入文檔圖像），(ii) 感知語言任務，包括多模態對話、圖像字幕、視覺問答，以及 (iii) 視覺任務，例如帶有描述的圖像識別（通過文本指令指定分類）。我們還表明，MLLM 可以從跨模式轉移中受益，即將知識從語言轉移到多模式，以及從多模式轉移到語言。


--------

### [Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models](https://paperswithcode.com/paper/visual-chatgpt-talking-drawing-and-editing)[2023年3月8日]
![](https://github.com/microsoft/TaskMatrix/raw/main/assets/demo_short.gif)
[microsoft/TaskMatrix](https://github.com/microsoft/visual-chatgpt)
#### 摘要
ChatGPT 吸引了跨領域的興趣，因為它提供了一個具有跨多個領域的卓越會話能力和推理能力的語言界面。然而，由於 ChatGPT 是用語言訓練的，因此它目前無法處理或生成來自視覺世界的圖像。與此同時，視覺基礎模型，例如視覺變壓器或穩定擴散，雖然表現出出色的視覺理解和生成能力，但它們只是具有一輪固定輸入和輸出的特定任務的專家。為此，我們構建了一個名為 **Visual ChatGPT** 的系統，包含不同的 Visual Foundation 模型，使用戶能夠通過以下方式與 ChatGPT 進行交互：
1) 不僅發送和接收語言，還發送和接收圖像
2) 提供複雜的視覺問題或視覺編輯指令，需要多個 AI 模型通過多個步驟進行協作。
3) 提供反饋並要求糾正結果。考慮到多個輸入/輸出的模型和需要視覺反饋的模型，我們設計了一系列提示將視覺模型信息注入ChatGPT。

實驗表明，Visual ChatGPT 為在 Visual Foundation 模型的幫助下研究 ChatGPT 的視覺角色打開了大門。

--------
### [Learning Text-Image Joint Embedding for Efficient Cross-Modal Retrieval with Deep Feature Engineering](https://paperswithcode.com/paper/learning-text-image-joint-embedding-for)[2021年10月22日]
![](https://github.com/git-disl/SEJE/raw/master/assets/generic.png)
[git-disl/seje](https://github.com/git-disl/seje)
#### 摘要
本文介紹了一種用於高效學習語義增強聯合嵌入的兩階段深度特徵工程框架，該框架將數據預處理中的深度特徵工程與訓練文本圖像聯合嵌入模型明確分開。我們使用 Recipe1M 數據集進行技術描述和實證驗證。在預處理中，我們通過將深度特徵工程與源自原始文本圖像輸入數據的語義上下文特徵相結合來執行深度特徵工程。我們利用 LSTM 識別關鍵術語、BERT 系列的深度 NLP 模型、TextRank 或 TF-IDF 來生成關鍵術語的排名分數，然後使用 word2vec 生成每個關鍵術語的向量表示。我們利用 WideResNet50 和 word2vec 提取和編碼食物圖像的圖像類別語義，以幫助聯合潛在空間中學習的食譜和圖像嵌入的語義對齊。在聯合嵌入學習中，我們通過使用軟邊緣和雙負採樣優化批處理硬三元組損失函數來執行深度特徵工程，同時還考慮到基於類別的對齊損失和基於鑑別器的對齊損失。大量實驗表明，我們採用深度特徵工程的 SEJE 方法明顯優於最先進的方法。還考慮基於類別的對齊損失和基於鑑別器的對齊損失。大量實驗表明，我們採用深度特徵工程的 SEJE 方法明顯優於最先進的方法。還考慮基於類別的對齊損失和基於鑑別器的對齊損失。大量實驗表明，我們採用深度特徵工程的 SEJE 方法明顯優於最先進的方法。

--------

### [ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions](https://paperswithcode.com/paper/chatgpt-asks-blip-2-answers-automatic)
![](https://github.com/Vision-CAIR/ChatCaptioner/raw/main/ChatCaptioner/demo_pic/demo1.gif)
[vision-cair/chatcaptioner](https://github.com/vision-cair/chatcaptioner)
#### 摘要
提出有洞察力的問題對於獲取知識和擴大我們對世界的理解至關重要。 然而，在人工智能研究中，提問的重要性在很大程度上被忽視了，模型主要是為了回答問題而開發的。 隨著 ChatGPT 等大型語言模型 (LLM) 的最新進展，我們發現它們能夠在提供合適的提示時提出高質量的問題。 這一發現為開發自動提問系統提供了新的機會。 在本文中，我們介紹了 ChatCaptioner，一種應用於圖像字幕的新型自動提問方法。 在這裡，ChatGPT 被提示向 BLIP-2（一種強大的視覺問答模型）提出一系列有關圖像的信息性問題。 通過不斷從 BLIP-2 的答案中獲取新的視覺信息，ChatCaptioner 能夠生成更豐富的圖像描述。 我們對 COCO、Conceptual Caption 和 WikiArt 等常見圖像字幕數據集進行人類受試者評估，並將 ChatCaptioner 與 BLIP-2 以及 ground Truth 進行比較。 我們的結果表明，ChatCaptioner 的字幕信息量顯著增加，人類評估者對提供最多圖像信息的投票數是其三倍。 此外，通過 WordNet 同義詞集匹配測量，ChatCaptioner 識別出的圖像中的對像比單獨使用 BLIP-2 多出 53%。

--------

## Text-to-Motion(這個有夠帥)
### [MotionGPT: Human Motion as a Foreign Language](https://paperswithcode.com/paper/motiongpt-human-motion-as-a-foreign-language)[2023年6月26日]
![](https://user-images.githubusercontent.com/16475892/247035914-5c7c455a-87c1-4b7e-b1e6-9e9433143e57.png)
[openmotionlab/motiongpt](https://github.com/openmotionlab/motiongpt)
#### 摘要
儘管預訓練大型語言模型不斷取得進展，但為語言和其他多模態數據（例如運動）構建統一模型的探索迄今為止仍然具有挑戰性且尚未觸及。幸運的是，人類運動表現出類似於人類語言的語義耦合，通常被視為肢體語言的一種形式。通過將語言數據與大規模運動模型融合，可以增強運動相關任務性能的運動語言預訓練變得可行。在這種見解的推動下，我們提出了 MotionGPT，這是一種統一、多功能且用戶友好的運動語言模型，用於處理多個與運動相關的任務。具體來說，我們對人體運動採用離散矢量量化，並將 3D 運動轉換為運動標記，類似於單詞標記的生成過程。在此“動作詞彙”的基礎上，我們以統一的方式對動作和文本進行語言建模，將人體動作視為一種特定的語言。此外，受即時學習的啟發，我們使用運動語言數據的混合來預訓練 MotionGPT，並根據基於提示的問答任務對其進行微調。大量實驗表明，MotionGPT 在多個運動任務上實現了最先進的性能，包括文本驅動的運動生成、運動字幕、運動預測和中間運動。

---------

### [ViCo: Detail-Preserving Visual Condition for Personalized Text-to-Image Generation](https://paperswithcode.com/paper/vico-detail-preserving-visual-condition-for)[2023年6月1日]
![](https://github.com/haoosz/ViCo/raw/main/img/teaser.png)
[haoosz/vico](https://github.com/haoosz/vico)
#### 摘要
最近提出了使用擴散模型的個性化文本到圖像生成並引起了廣泛關注。給定一些包含新穎概念的圖像（例如，獨特的玩具），我們的目標是調整生成模型以捕獲新穎概念的精細視覺細節，並根據文本條件生成逼真的圖像。我們提出了一種名為 ViCo 的插件方法，用於快速、輕量級的個性化生成。具體來說，我們提出了一個圖像注意模塊來根據補丁視覺語義來調節擴散過程。我們引入了一種基於注意力的對象掩模，它幾乎不需要注意力模塊的任何成本。此外，我們根據文本圖像注意力圖的內在屬性設計了一個簡單的正則化，以減輕常見的過度擬合退化。與許多現有模型不同，我們的方法不會微調原始擴散模型的任何參數。這允許更靈活和可轉移的模型部署。僅通過輕參數訓練（約擴散 U-Net 的 6%），我們的方法在定性和定量上就達到了與所有最先進模型相當甚至更好的性能。

----------


## 影片理解
### [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://paperswithcode.com/paper/video-chatgpt-towards-detailed-video)[2023年6月8日]
![](https://github.com/mbzuai-oryx/Video-ChatGPT/raw/main/docs/images/Video-ChatGPT.gif)
[mbzuai-oryx/Video-ChatGPT](https://github.com/mbzuai-oryx/video-chatgpt)
[Demo](https://www.ival-mbzuai.com/video-chatgpt)
#### 摘要
由大型語言模型 (LLM) 推動的對話代理正在提供一種與視覺數據交互的新方式。雖然已經對基於圖像的對話模型進行了初步嘗試，但這項工作通過引入 Video-ChatGPT 解決了基於視頻的對話領域尚未開發的問題。它是一個多模態模型，將視頻適應的視覺編碼器與法學碩士相結合。該模型能夠理解並生成類似人類的視頻對話。我們引入了一個包含 100,000 個視頻指令對的新數據集，用於訓練通過手動和半自動管道獲取的 Video-ChatGPT，該數據集易於擴展且對標籤噪聲具有魯棒性。我們還為基於視頻的對話模型開發了一個定量評估框架，以客觀地分析所提出模型的優點和缺點

--------

## 協作型Multimodal(以LLM為Assistant，調用模型或資料)
### [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://paperswithcode.com/paper/hugginggpt-solving-ai-tasks-with-chatgpt-and)[2023年3月30日]
![](https://github.com/microsoft/JARVIS/raw/main/assets/overview.jpg)
[microsoft/JARVIS](https://github.com/microsoft/JARVIS)
#### 摘要
利用不同領域和模式的人工智能（AI）模型解決複雜的AI任務，是實現人工通用智能的重要一步。儘管現有的AI模型在不同領域和模式上都很豐富，但它們無法應對複雜的AI任務。考慮到大型語言模型（LLM）在語言理解、生成、互動和推理方面展現出卓越的能力，我們主張LLM可以作為控制器來管理現有的AI模型，以解決複雜的AI任務，而語言可以作為一個通用的接口來賦予它們能力。基於這一理念，我們提出了HuggingGPT框架，該框架利用LLM（例如ChatGPT）將機器學習社區（例如Hugging Face）中的各種AI模型連接起來解決AI任務。具體而言，當接收到用戶請求時，我們使用ChatGPT進行任務規劃，根據Hugging Face中可用的模型功能描述選擇模型，使用選定的AI模型執行每個子任務，並根據執行結果進行回應摘要。通過充分利用ChatGPT的強大語言能力和Hugging Face中豐富的AI模型，HuggingGPT能夠涵蓋不同模式和領域中的眾多複雜AI任務，在語言、視覺、語音和其他具有挑戰性的任務中取得令人印象深刻的結果，為實現人工通用智能開辟了新的道路。

--------

### [Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models](https://paperswithcode.com/paper/chameleon-plug-and-play-compositional)[2023年4月19日]
![](https://github.com/lupantech/chameleon-llm/raw/main/assets/showcase_scienceqa.png)
[lupantech/chameleon-llm](https://github.com/lupantech/chameleon-llm)
#### 摘要
由於新興的推理能力，大型語言模型（LLM）在解決各種自然語言處理任務方面取得了顯著的進步。 然而，法學碩士有其固有的局限性，因為它們無法訪問最新信息（存儲在網絡上或特定任務的知識庫中）、使用外部工具以及執行精確的數學和邏輯推理。 在本文中，我們提出了 Chameleon，這是一種人工智能係統，它通過使用用於組合推理的即插即用模塊來增強法學碩士，從而緩解這些限制。 Chameleon 通過組合各種工具（例如 LLM、現成的視覺模型、網絡搜索引擎、Python 函數和基於啟發式的模塊）來綜合程序，以完成複雜的推理任務。 Chameleon 的核心是一個基於 LLM 的規劃器，它組裝了一系列工具來執行以生成最終響應。 我們展示了 Chameleon 在兩個多模態知識密集型推理任務上的有效性：ScienceQA 和 TabMWP。 Chameleon 由 GPT-4 提供支持，在 ScienceQA 上實現了 86.54% 的整體準確率，將最佳已發表的少數結果提高了 11.37%。 在 TabMWP 上，GPT-4 驅動的 Chameleon 將準確率提高了 17.0%，將最先進的準確率提升至 98.78%。 我們的分析還表明，與 ChatGPT 支持的規劃器相比，GPT-4 支持的規劃器通過從指令推斷潛在約束來表現出更加一致和合理的工具選擇。

--------

### [InternGPT: Solving Vision-Centric Tasks by Interacting with ChatGPT Beyond Language](https://paperswithcode.com/paper/internchat-solving-vision-centric-tasks-by)[2023年5月9日]
![](https://github.com/OpenGVLab/InternGPT/raw/main/assets/arch1.png)
[opengvlab/interngpt](https://github.com/opengvlab/interngpt)
#### 摘要
我們提出了一個名為 InternGPT 的交互式視覺框架，簡稱 iGPT。該框架將具有規劃和推理功能的聊天機器人（例如 ChatGPT）與非語言指令（例如指向動作）集成在一起，使用戶能夠直接操作屏幕上的圖像或視頻。指向（包括手勢、光標等）移動可以在執行需要細粒度控制、編輯和生成視覺內容的以視覺為中心的任務時提供更大的靈活性和精確度。InternGPT 。與現有依賴純語言的交互系統不同，通過結合指向指令，所提出的 iGPT 顯著提高了用戶和聊天機器人之間的通信效率，以及聊天機器人在以視覺為中心的任務中的準確性，特別是在對像數量大於2的複雜視覺場景中。此外，在iGPT中，使用輔助控制機制來提高LLM的控制能力，並且大量的稱為 Husky 的視覺語言模型針對高質量多模式對話進行了微調（令人印象深刻的是 ChatGPT-3.5-turbo 的 GPT-4 質量為 93.89%）。

--------

### [Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow](https://paperswithcode.com/paper/data-copilot-bridging-billions-of-data-and)[2023年6月12日]
![](https://github.com/zwq2018/Data-Copilot/raw/main/assets/fig1.png =70%x)
![](https://github.com/zwq2018/Data-Copilot/raw/main/demo1.png =70%x)
[zwq2018/data-copilot](https://github.com/zwq2018/data-copilot)
#### 摘要
金融、氣象、能源等各個行業每天都會產生海量的異構數據。 人類對有效管理、處理和顯示數據有著天然的需求。 然而，這些與數據相關的任務需要勞動密集型的工作和高水平的專業知識。 考慮到大型語言模型（LLM）在語義理解和推理方面表現出了良好的能力，我們主張部署LLM可以自主管理和處理大量數據，同時以人性化的方式顯示和交互。 基於這一信念，我們提出了 Data-Copilot，這是一種基於法學碩士的系統，它在一端連接眾多數據源，在另一端滿足不同的人類需求。 Data-Copilot 就像經驗豐富的專家一樣，自動將原始數據轉換為最符合用戶意圖的可視化結果。 具體來說，Data-Copilot 自主設計用於數據管理、處理、預測和可視化的多功能接口（工具）。 在實時響應中，針對用戶的請求，通過逐步調用相應接口，自動部署簡潔的工作流程。 界面設計和部署過程完全由Data-Copilot本身控制，無需人工協助。 此外，我們還打造了Data-Copilot demo，鏈接不同領域（股票、基金、公司、經濟、新聞直播）的海量數據，精準響應多樣化需求，成為可靠的AI助手。


-----------

# 影像模型
## Image Manipulation
### [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://paperswithcode.com/paper/drag-your-gan-interactive-point-based)[2023年5月18日]
![](https://github.com/XingangPan/DragGAN/blob/main/DragGAN.gif?raw=true)
[XingangPan/DragGAN](https://github.com/XingangPan/DragGAN)
[opengvlab/interngpt](https://github.com/opengvlab/interngpt)
#### 摘要
合成滿足用戶需求的視覺內容通常需要對生成對象的姿勢、形狀、表情和佈局進行靈活而精確的控制。現有方法通過手動註釋的訓練數據或先前的 3D 模型來獲得生成對抗網絡 (GAN) 的可控性，但這些方法通常缺乏靈活性、精確性和通用性。在這項工作中，我們研究了一種強大但較少探索的控制 GAN 的方法，即以用戶交互的方式“拖動”圖像的任意點以精確到達目標點，如圖 1 所示。為了實現這一目標，我們提出了 DragGAN，它由兩個主要組件組成：1）基於特徵的運動監督，驅動手柄點向目標位置移動，2）一種新的點跟踪方法，利用判別性生成器特徵來保持手柄點位置的本地化。通過 DragGAN，任何人都可以通過精確控制像素的去向來使圖像變形，從而操縱不同類別（例如動物、汽車、人類、風景等）的姿勢、形狀、表情和佈局。由於 GAN 的生成圖像流形，即使在具有挑戰性的場景中，它們也傾向於產生逼真的輸出，例如幻覺被遮擋的內容和始終遵循對像剛性的變形形狀。定性和定量比較都證明了 DragGAN 在圖像處理和點跟踪任務中相對於現有方法的優勢。

----------

## Segment Anything
### [Fast Segment Anything](https://paperswithcode.com/paper/fast-segment-anything)[2023年6月21日]
![](https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/assets/head_fig.png?raw=true)
[casia-iva-lab/fastsam](https://github.com/casia-iva-lab/fastsam)
#### 摘要
最近提出的分段任意模型（SAM）對許多計算機視覺任務產生了重大影響。它正在成為許多高級任務的基礎步驟，例如圖像分割、圖像標題和圖像編輯。但其巨大的計算成本使其無法在行業場景中得到更廣泛的應用。計算主要來自高分辨率輸入的 Transformer 架構。在本文中，我們針對這一基本任務提出了一種具有可比性能的加速替代方法。通過將任務重新表述為片段生成和提示，我們發現具有實例分割分支的常規 CNN 檢測器也可以很好地完成此任務。具體來說，我們將此任務轉換為經過充分研究的實例分割任務，並僅使用 SAM 作者發布的 SA-1B 數據集的 1/50 直接訓練現有的實例分割方法。通過我們的方法，我們實現了與 SAM 方法相當的性能，但運行速度提高了 50 倍。我們給出了足夠的實驗結果來證明其有效性。

-------
相信我，上下兩篇Paper有仇
### [Faster Segment Anything: Towards Lightweight SAM for Mobile Applications](https://paperswithcode.com/paper/faster-segment-anything-towards-lightweight)[2023年6月25日]
![](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/model_diagram.jpg)
[chaoningzhang/mobilesam](https://github.com/chaoningzhang/mobilesam)
#### 摘要
分割任何模型（SAM）是一種即時引導的視覺基礎模型，用於從背景中切出感興趣的對象。自從 Meta 研究團隊發布 SA 項目以來，SAM 因其令人印象深刻的零樣本傳輸性能以及與其他模型兼容的高級視覺應用（例如具有細粒度控制的圖像編輯）的多功能性而引起了廣泛關注。許多此類用例需要在資源受限的邊緣設備上運行，例如移動應用程序。在這項工作中，我們的目標是通過用輕量級圖像編碼器替換重量級圖像編碼器來使 SAM 適合移動設備。像原始 SAM 論文中那樣訓練這種新 SAM 的簡單方法會導致性能不令人滿意，尤其是在可用的訓練源有限的情況下。我們發現這主要是由圖像編碼器和掩模解碼器的耦合優化引起的，受此啟發，我們提出了解耦蒸餾。具體來說，我們將原始 SAM 中圖像編碼器 ViT-H 的知識提煉為輕量級圖像編碼器，它可以自動兼容原始 SAM 中的掩模解碼器。訓練可以在不到一天的時間內在單個 GPU 上完成，由此產生的輕量級 SAM 被稱為 MobileSAM，它的尺寸縮小了 60 多倍，但性能與原始 SAM 相當。對於推理速度，MobileSAM 每張圖像的運行時間約為 10 毫秒：圖像編碼器上為 8 毫秒，掩模解碼器上為 2 毫秒。憑藉卓越的性能和更高的通用性，我們的 MobileSAM 比並發 FastSAM 體積小 7 倍，速度快 4 倍，更適合移動應用。

---------

### [Generative Semantic Segmentation](https://paperswithcode.com/paper/generative-semantic-segmentation)[CVPR 2023]
![](https://hackmd.io/_uploads/ryx9ZjGFn.png)
[fudan-zvg/gss](https://github.com/fudan-zvg/gss)

#### 摘要
我們提出了生成語義分割（GSS），一種用於語義分割的生成學習方法。獨特的是，我們將語義分割視為圖像條件掩模生成問題。這是通過用潛在的先驗學習過程取代傳統的每像素判別學習來實現的。具體來說，我們對給定分割掩模的潛在變量的變分後驗分佈進行建模。為此，分割掩模用一種特殊類型的圖像（稱為掩模）來表示。這種後驗分佈允許無條件地生成分割掩模。為了實現給定圖像的語義分割，我們進一步引入了調節網絡。它是通過最小化 maskige 後驗分佈之間的分歧來優化的（即 分割掩模）和輸入訓練圖像的潛在先驗分佈。對標準基準的大量實驗表明，我們的 GSS 可以在標準語義分割設置中與現有技術替代方案競爭，同時在更具挑戰性的跨域設置中實現新的技術水平。


---------

## Feature Matcher
### [LightGlue: Local Feature Matching at Light Speed](https://paperswithcode.com/paper/lightglue-local-feature-matching-at-light)[2023年6月23日]
![](https://github.com/cvg/LightGlue/raw/main/assets/easy_hard.jpg)
[cvg/lightglue](https://github.com/cvg/lightglue#lightgluelocal-feature-matching-at-light-speed)
#### 摘要
介紹 LightGlue，這是一種深度神經網絡，可以學習匹配圖像之間的局部特徵。我們重新審視 SuperGlue 的多項設計決策（稀疏匹配領域的最新技術），並得出簡單但有效的改進。累積起來，它們使 LightGlue 更加高效——在內存和計算方面，更加準確，並且更容易訓練。LightGlue 的一個關鍵特性是它能夠適應問題的難度：對於直觀上易於匹配的圖像對（例如，由於較大的視覺重疊或有限的外觀變化），推理速度要快得多。這為在 3D 重建等延遲敏感的應用程序中部署深度匹配器開闢了令人興奮的前景。

---------

## Track Object
### [Track Anything: Segment Anything Meets Videos](https://paperswithcode.com/paper/track-anything-segment-anything-meets-videos)[2023年4月24日]
![](https://github.com/gaomingqi/Track-Anything/raw/master/assets/avengers.gif)
[gaomingqi/Track-Anything](https://github.com/gaomingqi/track-anything)
#### 摘要
最近，分段任意模型（SAM）由於其令人印象深刻的圖像分割性能而迅速受到廣泛關注。由於其強大的圖像分割能力以及不同提示的高交互性，我們發現它在視頻的一致分割方面表現不佳。因此，在本報告中，我們提出了 Track Anything Model (TAM)，它實現了視頻中的高性能交互式跟踪和分割。詳細來說，給定一個視頻序列，只需很少的人類參與，即幾次點擊，人們就可以跟踪他們感興趣的任何內容，並在一次推理中獲得滿意的結果。無需額外訓練，這種交互式設計在視頻對象跟踪和分割方面的表現令人印象深刻。

---------

# LLM(Large Language Model)
## NER相關
### [MFE-NER: Multi-feature Fusion Embedding for Chinese Named Entity Recognition](https://paperswithcode.com/paper/mfe-ner-multi-feature-fusion-embedding-for)[2021年9月16日]
![](https://hackmd.io/_uploads/BkdaD5fF2.png)

#### 摘要
預訓練的語言模型將命名實體識別（NER）帶入了一個新時代，但需要更多的知識來提高其在特定問題上的性能。在本文中，我們提出了一種新方法，即中文命名實體識別的多特徵融合嵌入（MFE-NER），以強化中文的語言模式並處理中文命名實體識別中的字符替換問題。MFE 將語義、字形和語音特徵融合在一起。在字形域中，我們將漢字拆解為組件來表示結構特徵，使得具有相似結構的字符可以具有緊密的嵌入空間表示。同時，我們的工作還提出了一種改進的語音系統，使得漢字之間的語音相似度計算更加合理。

---------

## 不同訓練方法
### [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://paperswithcode.com/paper/tree-of-thoughts-deliberate-problem-solving)[2023年5月17日]
![](https://github.com/princeton-nlp/tree-of-thought-llm/raw/master/pics/teaser.png)
[princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm)
[kyegomez/tree-of-thoughts](https://github.com/kyegomez/tree-of-thoughts)
#### 摘要
語言模型越來越多地被部署用於解決各種任務中的一般問題，但在推理過程中仍然僅限於Token層級、從左到右的決策過程。這意味著他們可能無法完成需要探索、戰略前瞻性或初始決策發揮關鍵作用的任務。為了克服這些挑戰，我們引入了一種新的語言模型推理框架——思想樹（ToT），它概括了流行的思想鏈方法來提示語言模型，並能夠探索作為文本（思想）的連貫單元。解決問題的中間步驟。ToT 允許 LM 通過考慮多種不同的推理路徑和自我評估選擇來執行深思熟慮的決策，以決定下一步的行動方案，以及在必要時展望未來或回溯以做出全球選擇。我們的實驗表明，ToT 顯著增強了語言模型在三個需要重要規劃或搜索的新任務上解決問題的能力：24 人遊戲、創意寫作和迷你填字遊戲。

------------

### [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://paperswithcode.com/paper/fine-tuning-language-models-with-just-forward)[2023年5月27日]
![](https://raw.githubusercontent.com/princeton-nlp/MeZO/main/assets/fig2.png)
#### 摘要
在本文中，我們提出了一種內存高效的零階優化器（MeZO），採用經典的零階 SGD 方法進行就地操作，從而微調語言模型（LM），其內存佔用與推理相同。
使用單個 A100 80GB GPU，MeZO 可以訓練 300 億參數的 OPT 模型，而使用 Adam 進行微調只能訓練 2.7B LM。MeZO 表現出與跨多個任務的反向傳播微調相當的性能，內存減少高達 12 倍。MeZO 還兼容全參數和參數高效調整技術，例如 LoRA 和前綴調整。我們還表明 MeZO 可以有效地優化不可微目標（例如，最大化準確性或 F1）。

---------

### [BayLing: Bridging Cross-lingual Alignment and Instruction Following through Interactive Translation for Large Language Models](https://paperswithcode.com/paper/bayling-bridging-cross-lingual-alignment-and)[2023年6月19日]
![](https://github.com/ictnlp/BayLing/raw/main/assets/gui.gif)
[ictnlp/BayLing](https://github.com/ictnlp/bayling)
[Demo](http://nlp.ict.ac.cn/bayling/demo)
#### 摘要
大型語言模型（LLM）在語言理解和生成方面表現出了非凡的能力。為了最大限度地減少人力工作量，我們建議通過交互式翻譯任務將語言生成和指令跟踪的能力從英語轉移到其他語言。我們開發了BayLing，以LLaMA為基礎的LLM，自動構建交互式翻譯指令用於指導調優的指令跟隨LLM。廣泛的評估表明，BayLing 實現了與 GPT-3.5-turbo 相當的性能，儘管使用的參數大小要小得多，僅為 130 億。翻譯任務的實驗結果表明，與自動評估的 GPT-4 相比，BayLing 實現了 95% 的單輪翻譯能力；與人工評估的 GPT-3.5-turbo 相比，BayLing 實現了 96% 的交互式翻譯能力。為了評估一般任務的性能，我們創建了一個名為 BayLing-80 的多輪指令測試集。BayLing-80 上的實驗結果表明，與 GPT-3.5-turbo 相比，BayLing 實現了 89% 的性能。

----------

### [Offsite-Tuning: Transfer Learning without Full Model](https://paperswithcode.com/paper/offsite-tuning-transfer-learning-without-full)[2023年2月9日]
![](https://github.com/mit-han-lab/offsite-tuning/raw/main/figures/overview.png)
[mit-han-lab/offsite-tuning](https://github.com/mit-han-lab/offsite-tuning)

#### 摘要
遷移學習對於基礎模型適應下游任務非常重要。然而，許多基礎模型都是專有的，因此用戶必須與模型所有者共享數據來微調模型，這不僅成本高昂，而且會引發隱私問題。此外，對大型基礎模型進行微調對於大多數下游用戶來說是計算密集型的且不切實際的。在本文中，我們提出了 Offsite-Tuning，這是一種隱私保護且高效的遷移學習框架，可以使數十億參數的基礎模型適應下游數據，而無需訪問完整模型。在場外調整中，模型所有者向數據所有者發送一個輕量級適配器和一個有損壓縮模擬器，然後數據所有者在模擬器的幫助下對下游數據上的適配器進行微調。然後將微調後的適配器返回給模型所有者，誰將其插入完整模型以創建適應的基礎模型。異地調整可以保護雙方的隱私，並且在計算上比需要訪問完整模型權重的現有微調方法更有效。我們展示了在各種大型語言和視覺基礎模型上進行異地調整的有效性。場外調優可以達到與全模型微調相當的速度，同時保護隱私且高效，實現 6.5 倍的加速和 5.6 倍的內存減少。

------

### [Let's Verify Step by Step](https://paperswithcode.com/paper/let-s-verify-step-by-step-1)[Preprint 2023]
![](https://hackmd.io/_uploads/SyxjsoMYh.png)
[openai/prm800k](https://github.com/openai/prm800k)
#### 摘要
近年來，大型語言模型在執行複雜的多步驟推理的能力方面有了很大的提高。然而，即使是最先進的模型仍然經常產生邏輯錯誤。為了訓練更可靠的模型，我們可以轉向結果監督（為最終結果提供反饋）或過程監督（為每個中間推理步驟提供反饋）。考慮到訓練可靠模型的重要性，以及人類反饋的高昂成本，仔細比較這兩種方法非常重要。最近的工作已經開始進行這種比較，但仍然存在許多問題。我們進行了自己的調查，發現在解決具有挑戰性的數學數據集中的問題時，過程監督明顯優於訓練模型的結果監督。我們的過程監督模型解決了 MATH 測試集代表性子集中 78% 的問題。此外，我們表明主動學習顯著提高了過程監督的效率。為了支持相關研究，我們還發布了 PRM800K，這是一個包含 800,000 個步驟級人類反饋標籤的完整數據集，用於訓練我們的最佳獎勵模型。

------

### [A Simple and Effective Pruning Approach for Large Language Models](https://paperswithcode.com/paper/a-simple-and-effective-pruning-approach-for)[2023年6月20日]
![](https://user-images.githubusercontent.com/20168304/245999360-f951de47-269d-491d-826a-8e6d85627849.png)
[locuslab/wanda](https://github.com/locuslab/wanda)
#### 摘要
隨著規模的增加，大型語言模型（LLM）自然成為網絡修剪方法的候選者：在努力保持性能的同時刪除網絡權重子集的方法。然而，現有方法要么需要重新訓練，這對於數十億規模的法學碩士來說很少負擔得起，要么需要解決依賴於二階信息的權重重建問題，這也可能在計算上昂貴。在本文中，我們介紹了一種新穎、簡單而有效的剪枝方法，稱為 Wanda（按權重和激活進行剪枝），旨在誘導預訓練的 LLM 的稀疏性。受最近對法學碩士中出現的大量特徵的觀察的啟發，我們的方法在每個輸出的基礎上修剪最小量值乘以相應的輸入激活的權重。尤其，Wanda不需要重新訓練或權重更新，修剪後的LLM可以按原樣使用。我們在 LLaMA 上跨各種語言基準對我們的方法進行了徹底的評估。

------------

### [Thought Cloning: Learning to Think while Acting by Imitating Human Thinking](https://paperswithcode.com/paper/thought-cloning-learning-to-think-while)[2023年6月1日]
![](https://github.com/ShengranHu/Thought-Cloning/raw/main/media/TC_framework.png)
[ShengranHu/Thought-Cloning](https://github.com/ShengranHu/Thought-Cloning)
#### 摘要
語言通常被認為是人類思維的一個關鍵方面，為我們提供了概括、探索、計劃、重新計劃和適應新情況的卓越能力。然而，強化學習（RL）代理在這些能力中的任何一個都遠未達到人類水平。 我們假設這種認知缺陷的原因之一是它們缺乏語言思維的好處，並且我們可以通過訓練它們像人類一樣思考來改進人工智能代理。 我們引入了一種新穎的模仿學習框架，即思想克隆，其想法不僅是克隆人類示威者的行為，而且還克隆人類在執行這些行為時的想法。 雖然我們期望思維克隆能夠在人類在行動時大聲思考的互聯網規模數據集（例如帶有文字記錄的在線視頻）上真正大規模發揮作用，但在這裡我們在綜合生成思維和行動數據的領域進行實驗。結果表明，思想克隆的學習速度比行為克隆快得多，並且其性能優勢隨著測試任務的分佈程度的增加而增加，突顯了其更好地處理新情況的能力。 思維克隆還為人工智能安全性和可解釋性提供了重要的好處，並使調試和改進人工智能變得更加容易。 因為我們可以觀察智能體的想法，所以我們可以（1）更容易地診斷為什麼會出現問題，從而更容易解決問題，（2）通過糾正智能體的思維來引導智能體，或者（3）防止它做不安全的事情 它計劃做的事情。

------------

### [Multimodal Chain-of-Thought Reasoning in Language Models](https://paperswithcode.com/paper/multimodal-chain-of-thought-reasoning-in)[2023年2月2日]
![](https://github.com/amazon-science/mm-cot/raw/main/vision_features/mm-cot.png)
[amazon-science/mm-cot](https://github.com/amazon-science/mm-cot)
[xqx12/daily-info](https://github.com/xqx12/daily-info)
#### 摘要
大型語言模型 (LLM) 通過利用思維鏈 (CoT) 提示生成中間推理鏈作為推斷答案的基本原理，在復雜推理方面表現出了令人印象深刻的性能。然而，現有的 CoT 研究主要集中在語言情態上。我們提出了 Multimodal-CoT，它將語言（文本）和視覺（圖像）模態合併到一個兩階段框架中，將基本原理生成和答案推理分開。通過這種方式，答案推理可以更好地利用基於多模態信息生成的基本原理。借助 Multimodal-CoT，我們的模型在 10 億個參數下，在 ScienceQA 基准上的表現比之前最先進的 LLM (GPT-3.5) 提高了 16 個百分點（75.17%->91.68% 準確率），甚至超越了人類表現。

-----------

## 不同應用(無限長度)
### [Unlimiformer: Long-Range Transformers with Unlimited Length Input](https://paperswithcode.com/paper/unlimiformer-long-range-transformers-with)[2023年5月2日]
![](https://user-images.githubusercontent.com/42593540/236538293-1d5fdfe3-3e34-4979-9611-a9c9f56e3a00.png)
[abertsch72/unlimiformer](https://github.com/abertsch72/unlimiformer)
#### 摘要
自從變壓器的提出以來，這些模型就被限制在有界的輸入長度，因為它們需要關注輸入中的每個標記。在這項工作中，我們提出了 Unlimiformer：一種通用方法，它包裝任何現有的預訓練編碼器-解碼器轉換器，並將交叉注意力計算卸載到單個 k 最近鄰 (kNN) 索引，而返回的 kNN 距離是注意力點-產品分數。該 kNN 索引可以保存在 GPU 或 CPU 內存上，並在亞線性時間內進行查詢；這樣，我們可以索引幾乎無限的輸入序列，而每個解碼器層中的每個注意力頭都會檢索其前 k 個鍵，而不是關注每個鍵。我們根據幾個長文檔和書籍摘要基準評估 Unlimiformer，顯示它甚至可以處理來自 BookSum 數據集的 500k 令牌長輸入，並且在測試時不會進行任何輸入截斷。

------------

# Loss Function
## 圖片分類
### [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://paperswithcode.com/paper/arcface-additive-angular-margin-loss-for-deep)[CVPR 2019]
![](https://camo.githubusercontent.com/94afea3fd9149d1859c601e5f396b2d57ed803d8cc1401fcbdc76e669e162943/68747470733a2f2f696e7369676874666163652e61692f6173736574732f696d672f6769746875622f666163657265636f676e6974696f6e66726f6d766964656f2e504e47)
![](https://chtseng.files.wordpress.com/2022/09/null-21.png)
<font size=6>$L_{3} = -\frac{1}{N}\sum^{N}_{i=1}\log\frac{e^{s\left(\cos\left(\theta_{y_{i}} + m\right)\right)}}{e^{s\left(\cos\left(\theta_{y_{i}} + m\right)\right)} + \sum^{n}_{j=1, j \neq y_{i}}e^{s\cos\theta_{j}}}$</font><br>


#### 摘要
在本文中，我們首先引入了Additive Angular Margin Loss（ArcFace），它不僅具有清晰的幾何解釋，而且顯著增強了判別力。由於ArcFace容易受到**大量Label Noise**的影響，我們進一步提出了子中心ArcFace，其中每個類都包含K子中心和訓練樣本只需靠近任意一個K積極的分中心。子中心 ArcFace 鼓勵包含大多數乾淨面孔的主導子類和包含堅硬或嘈雜面孔的非主導子類。基於這種自行式隔離，我們通過在大量現實世界噪聲下自動淨化原始網頁面來提高性能。除了判別性特徵嵌入之外，我們還探索逆問題，將特徵向量映射到人臉圖像。無需訓練任何額外的生成器或判別器，預訓練的 ArcFace 模型只需使用網絡梯度和批量歸一化 (BN) 先驗，即可為訓練數據內部和外部的受試者生成身份保留的人臉圖像。大量實驗表明，ArcFace 可以增強判別性特徵嵌入並增強生成人臉合成。

---------

# Survey Paper
## Multimodal
### [A Survey on Multimodal Large Language Models](https://paperswithcode.com/paper/a-survey-on-multimodal-large-language-models)[2023年6月23日]
![](https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/main/images/xmind.png)
#### 摘要
多模態大語言模型（MLLM）最近成為一個新的研究熱點，它使用強大的大語言模型（LLM）作為大腦來執行多模態任務。MLLM 令人驚訝的新興功能，例如基於圖像編寫故事和無 OCR 的數學推理，在傳統方法中很少見，這表明了通向通用人工智能的潛在途徑。本文旨在追溯和總結MLLM的最新進展。首先，我們提出了MLLM的表述並描述了其相關概念。然後，我們討論關鍵技術和應用，包括多模態指令調優（M-IT）、多模態上下文學習（M-ICL）、多模態思維鏈（M-CoT）和法學碩士輔助視覺推理（LAVR） 。最後，我們討論現有的挑戰並指出有前途的研究方向。鑑於MLLM時代才剛剛開始，我們將不斷更新這項調查，希望它能激發更多的研究。

---------

## Autonomous Agent
### [A Survey on Large Language Model based Autonomous Agents](https://paperswithcode.com/paper/a-survey-on-large-language-model-based)[2023年8月22]
![](https://hackmd.io/_uploads/r1nadsHTn.png)
[paitesanshi/llm-agent-survey](https://github.com/paitesanshi/llm-agent-survey)
#### 摘要
自主代理長期以來一直是學術界的一個突出研究課題。先前該領域的研究通常側重於在孤立的環境中訓練知識有限的智能體。在本文中，我們對這些研究進行了全面的調查，從整體角度對自主代理領域進行了系統回顧。 更具體地說，我們的重點在於構建基於 LLM 的代理，為此我們提出了一個包含大部分先前工作的統一框架。 此外，我們還總結了基於 LLM 的人工智能代理在社會科學、自然科學和工程領域的各種應用。 

---------

## ChatGPT
### [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)[2023年4月26日]
![](https://github.com/Mooler0410/LLMsPracticalGuide/raw/main/imgs/qr_version.jpg)
[Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)
#### 摘要
這篇Survey很有用，其中的Github料很多。
* 預訓練數據
    * RedPajama-Data: An Open Source Recipe to Reproduce LLaMA training dataset，2023。[Github](https://github.com/togethercomputer/RedPajama-Data)
        * Dataset	Token Count
            | Dataset | Token Count |
            | -------- | -------- |
            | Commoncrawl | 878 Billion |
            | C4 | 175 Billion |
            | GitHub | 59 Billion |
            | Books | 26 Billion |
            | ArXiv | 28 Billion |
            | Wikipedia | 24 Billion |
            | StackExchange | 20 Billion |
            | Total | 1.2 Trillion |
    * The Pile：用於語言建模的 800GB 多樣化文本數據集，Arxiv 2020。[論文](https://arxiv.org/abs/2101.00027)
    * 預訓練目標如何影響大型語言模型學習語言屬性的能力？，ACL 2022。[論文](https://aclanthology.org/2022.acl-short.16/)
    * 神經語言模型的縮放定律，2020。[論文](https://arxiv.org/abs/2001.08361)
    * 以數據為中心的人工智能：一項調查，2023 年。[論文](https://arxiv.org/abs/2303.10158)
    * GPT如何獲得它的能力？追踪語言模型的新興能力的來源，2022 年。[Blog](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
* 微調數據
    * 零樣本文本分類基準測試：數據集、評估和蘊含方法，EMNLP 2019。[論文](https://arxiv.org/abs/1909.00161)
    * 語言模型是少樣本學習者，NIPS 2020。[論文](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
    * 法學碩士的綜合數據生成有助於臨床文本挖掘嗎？Arxiv 2023[論文](https://arxiv.org/abs/2303.04360)
* 測試數據/用戶數據
    * 自然語言理解中大型語言模型的快捷學習：一項調查，Arxiv 2023。[論文](https://arxiv.org/abs/2208.11857)
    * 關於 ChatGPT 的魯棒性：對抗性和非分佈視角Arxiv，2023。[論文](https://arxiv.org/abs/2302.12095)
    * SuperGLUE：通用語言理解系統的更具粘性的基準Arxiv 2019。[論文](https://arxiv.org/abs/1905.00537)

-------

## LLM
### [A Survey of Large Language Models](https://paperswithcode.com/paper/a-survey-of-large-language-models)[2023年3月31日]
![](https://github.com/RUCAIBox/LLMSurvey/raw/main/assets/llama-0628-final.png)
![](https://github.com/RUCAIBox/LLMSurvey/raw/main/assets/gpt-series.png)
![](https://github.com/RUCAIBox/LLMSurvey/raw/main/assets/arxiv_llms.png)
[rucaibox/llmsurvey](https://github.com/rucaibox/llmsurvey)
#### 摘要
語言本質上是一個受語法規則控制的複雜的人類表達系統。開發強大的人工智能算法來理解和掌握語言提出了重大挑戰。作為一種主要方法，語言建模在過去二十年中被廣泛研究用於語言理解和生成，從統計語言模型發展到神經語言模型。最近，通過在大規模語料庫上預訓練 Transformer 模型提出了預訓練語言模型（PLM），在解決各種 NLP 任務方面表現出了強大的能力。由於研究人員發現模型縮放可以帶來性能提升，因此他們通過將模型大小增加到更大來進一步研究縮放效果。有趣的是，當參數規模超過一定水平時，這些擴大的語言模型不僅實現了顯著的性能提升，而且還表現出了一些小規模語言模型所不具備的特殊能力。為了區分參數規模的差異，研究界為具有顯著規模的 PLM 創造了術語“大語言模型 (LLM)”。近年來，學界和工業界對LLM的研究取得了較大進展，其中一個顯著進展是ChatGPT的推出，引起了社會的廣泛關注。法學碩士的技術發展對整個人工智能社區產生了重要影響，這將徹底改變我們開發和使用人工智能算法的方式。在本次調查中，我們通過介紹背景、主要發現和主流技術來回顧法學碩士的最新進展。尤其，我們重點關注LLM的四個主要方面，即預訓練、適應調優、利用和能力評估。此外，我們還總結了發展法學碩士的可用資源，並討論了未來方向的剩餘問題。

-------

## CV(Computer Vision)
### [Towards Open Vocabulary Learning: A Survey](https://paperswithcode.com/paper/towards-open-vocabulary-learning-a-survey)[2023年6月28日]
![](https://github.com/jianzongwu/Awesome-Open-Vocabulary/raw/main/figs/timeline.jpg)
[jianzongwu/awesome-open-vocabulary](https://github.com/jianzongwu/awesome-open-vocabulary)
#### 摘要
在視覺場景理解領域，深度神經網絡在分割、跟踪和檢測等各種核心任務中取得了令人印象深刻的進步。然而，大多數方法都基於閉集假設，這意味著模型只能識別訓練集中存在的預定義類別。最近，由於視覺語言預訓練的快速進展，開放詞彙設置被提出。這些新方法尋求定位和識別帶註釋的標籤空間之外的類別。與弱監督和零樣本設置相比，開放詞彙方法更通用、更實用、更有效。本文對開放詞彙學習進行了全面回顧，總結和分析了該領域的最新發展。尤其，我們首先將其與零樣本學習、開放集識別和分佈外檢測等相關概念進行比較。然後，我們回顧了分割和檢測中幾個密切相關的任務，包括長尾問題、少樣本和零樣本設置。對於方法綜述，我們首先介紹近距離檢測和分割的基本知識作為初步知識。接下來，我們研究使用開放詞彙學習的各種場景，確定常見的設計元素和核心思想。然後，我們比較常用數據集和基準中最新的檢測和分割方法。最後，我們總結了關於未來研究方向的見解、問題和討論。據我們所知，這是第一篇關於開放詞彙學習的全面文獻綜述。

----------

## Diffusion Model
### [Diffusion Models for Image Restoration and Enhancement -- A Comprehensive Survey](https://paperswithcode.com/paper/diffusion-models-for-image-restoration-and)[2023年8月18日]
![](https://hackmd.io/_uploads/ByW1_jBp2.png)
[lixinustc/awesome-diffusion-model-for-image-processing](https://github.com/lixinustc/awesome-diffusion-model-for-image-processing)
#### 摘要
圖像恢復（IR）一直是低水平視覺領域不可或缺的且具有挑戰性的任務，其努力提高因各種形式的退化而扭曲的圖像的主觀質量。在本文中，我們首次對最近基於擴散模型的圖像恢復方法進行了全面回顧，包括學習範式、條件策略、框架設計、建模策略和評估。 具體來說，我們首先簡要介紹擴散模型的背景，然後介紹在圖像恢復中利用擴散模型的兩種流行的工作流程。隨後，我們對紅外和盲/現實世界紅外使用擴散模型的創新設計進行了分類和強調，旨在激發未來的發展。 為了徹底評估現有方法，我們總結了常用的數據集、實現細節和評估指標。 此外，我們還對開源方法在圖像超分辨率、去模糊和修復等三個任務中進行了客觀比較。 最終，根據現有工作的局限性，我們為基於擴散模型的IR的未來研究提出了五個潛在和具有挑戰性的方向，包括採樣效率、模型壓縮、失真模擬和估計、失真不變學習和框架設計。

----------


# 訓練框架
## Self-Supervising Learning(SSL)
### [EMP-SSL: Towards Self-Supervised Learning in One Training Epoch
](https://paperswithcode.com/paper/emp-ssl-towards-self-supervised-learning-in)
![](https://hackmd.io/_uploads/rynyZ4Rih.png =80%x)
![](https://hackmd.io/_uploads/S1Mz-4Rs2.png =80%x)

#### 摘要
在這項工作中，我們表明高效自監督學習的關鍵是增加每個圖像實例的作物數量。利用最先進的 SSL 方法之一，我們引入了一種簡單形式的自我監督學習方法，稱為極限多補丁自我監督學習（EMP-SSL），該方法不依賴於許多啟發式技術SSL例如分支之間的權重共享、特徵歸一化、輸出量化和停止梯度等，並將訓練週期減少了兩個數量級。我們表明，所提出的方法僅在一個epoch 內就能在CIFAR-10 上收斂到85.1%，在CIFAR-100 上收斂到58.5%，在Tiny ImageNet 上收斂到38.1%，在ImageNet-100 上收斂到58.5%。此外，該方法在不到10 個訓練週期內進行線性探測，在CIFAR-10 上實現了91.5%，在CIFAR-100 上實現了70.1%，在Tiny ImageNet 上實現了51.5%，在ImageNet-100 上實現了78.9%。此外，我們還發現，與基線 SSL 方法相比，EMP-SSL 對域外數據集的可轉移性明顯更好。我們將在 https://github.com/tsb0601/EMP-SSL 中發布代碼。在 ImageNet-100 上，在不到 10 個訓練週期內進行線性探測，達到 9%。此外，我們還發現，與基線 SSL 方法相比，EMP-SSL 對域外數據集的可轉移性明顯更好。

------------

## 針對精度
### [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://paperswithcode.com/paper/llm-int8-8-bit-matrix-multiplication-for)[2022年8月15日]
![](https://hackmd.io/_uploads/HyrpwjfY2.png)
![](https://hackmd.io/_uploads/HkMJuofY3.png =50%x)

[timdettmers/bitsandbytes](https://github.com/timdettmers/bitsandbytes)

#### 摘要
大型語言模型已被廣泛採用，但需要大量 GPU 內存進行推理。 我們為 Transformer 中的前饋和注意力投影層開發了一種 Int8 矩陣乘法程序，它將推理所需的內存減少了一半，同時保留了完整的精度性能。 使用我們的方法，可以加載 175B 參數 16/32 位檢查點，轉換為 Int8，並立即使用，而不會降低性能。 這是通過理解和解決 Transformer 語言模型中高度系統化的新興特徵的屬性來實現的，這些特徵主導了注意力和 Transformer 的預測性能。 為了應對這些特徵，我們開發了一個由兩部分組成的量化過程，LLM.int8()。 我們首先使用向量量化，對矩陣乘法中的每個內積使用單獨的歸一化常數，以量化大多數特徵。 然而，對於出現的離群值，我們還採用了一種新的混合精度分解方案，它將離群值特徵維度隔離為 16 位矩陣乘法，而仍然超過 99.9% 的值以 8 位相乘。 使用 LLM.int8()，我們憑經驗證明可以在具有多達 175B 個參數的 LLM 中執行推理，而不會降低任何性能。 這一結果使此類模型更易於訪問，例如可以在具有消費級 GPU 的單個服務器上使用 OPT-175B/BLOOM。 我們開源我們的軟件。

-------

## 通用框架(不管多少GPU都能用)
### [Colossal-Auto: Unified Automation of Parallelization and Activation Checkpoint for Large-scale Models](https://paperswithcode.com/paper/map-memory-aware-automated-intra-op-parallel)[2023年2月6日]
![](https://hackmd.io/_uploads/r1iH39Mth.png =70%x)

![](https://hackmd.io/_uploads/r1xwncGt3.png =70%x)

![](https://hackmd.io/_uploads/Sy3_3qGFn.png =70%x)

[hpcaitech/colossalai](https://github.com/hpcaitech/colossalai)
#### 摘要
近年來，大型模型在各個領域都展現出了最先進的性能。然而，訓練此類模型需要各種技術來解決 GPU 等設備上計算能力和內存有限的問題。一些常用的技術包括管道並行、張量並行和激活檢查點。雖然現有的工作主要集中在尋找高效的分佈式執行計劃和激活檢查點調度，但還沒有提出聯合優化這兩個計劃的方法。提前編譯嚴重依賴於準確的內存和計算開銷估計，這通常非常耗時且具有誤導性。現有的訓練系統和機器學習管道要么物理執行每個操作數，要么使用縮放的輸入張量估計內存使用情況。為了應對這些挑戰，我們引入了一個可以聯合優化分佈式執行和梯度檢查點計劃的系統。此外，我們還提供了一個易於使用的符號分析器，可以以最小的時間成本為任何 PyTorch 模型生成內存和計算統計數據。我們的方法允許用戶 在給定硬件上以最少的代碼更改並行化模型訓練。

--------

# 資料集
## 圖片生成
### [GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image](https://paperswithcode.com/paper/genimage-a-million-scale-benchmark-for)[2023年6月14日]
![](https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/Examples/visulization.png =80%x)
[genimage-dataset/genimage](https://github.com/genimage-dataset/genimage)
[andrew-zhu/genimage](https://github.com/andrew-zhu/genimage)
#### 摘要
生成模型生成攝影圖像的非凡能力加劇了人們對虛假信息傳播的擔憂，從而導致需要能夠區分人工智能生成的假圖像和真實圖像的探測器。然而，缺乏包含來自最先進圖像生成器的圖像的大型數據集對此類探測器的開發構成了障礙。在本文中，我們介紹了 GenImage 數據集，它具有以下優點：1）大量圖像，包括超過一百萬對 AI 生成的假圖像和收集的真實圖像。2）豐富的圖像內容，涵蓋廣泛的圖像類別。3) 最先進的生成器，利用先進的擴散模型和 GAN 合成圖像。上述優點使得在 GenImage 上訓練的檢測器能夠經過全面的評估，並表現出對不同圖像的強大適用性。我們對數據集進行了全面分析，並提出了兩個任務來評估類似於真實場景的檢測方法。跨生成器圖像分類任務測量在一個生成器上訓練的檢測器在其他生成器上進行測試時的性能。退化圖像分類任務評估檢測器處理低分辨率、模糊和壓縮圖像等退化圖像的能力。與主流方法相比，

---------

## 問答資料集(中文)
### [silk-road/Luotuo-QA-A-CoQA-Chinese](https://huggingface.co/datasets/silk-road/Luotuo-QA-A-CoQA-Chinese)[2023年5月]
![](https://hackmd.io/_uploads/B1jz95GFh.png)
#### 摘要
CoQA(Conversational Question Answering)數據集是一個用於對話問答任務的海量數據集，包含超過127,000個問題及其對應的答案。這些文本式來自七個不同領域的文章：兒童故事、文學作品、中學和高中英語考試、新聞、維基百科、Reddit和科學。由於這個數據集是我們Luotuo-QA項目的一部分，我們將其命名為luotuo-QA-A，旨在促進對話式問答在漢語環境下的研究和應用。您可以在這裡查看Luotuo-QA項目：https://github.com/LC1332/Luotuo-QA

---------

### [sunzeyeah/chinese_chatgpt_corpus](https://huggingface.co/datasets/sunzeyeah/chinese_chatgpt_corpus)[2023年3月]
![](https://hackmd.io/_uploads/Bk2AY5zK3.png)

#### 摘要
該Database收集了用於監督微調（SFT）和人類反饋強化學習（RLHF）的中文語料庫。
* 下載的數據集文件大小： 5.04 GB
* 生成的數據集大小： 0 GB
* 使用的Rom總量： 5.04 GB

## Vedeo-to-text
### [Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks](https://paperswithcode.com/paper/youku-mplug-a-10-million-large-scale-chinese)[2023年6月7日]
![](https://github.com/X-PLUG/Youku-mPLUG/raw/main/assets/case1.jpg)
![](https://github.com/X-PLUG/Youku-mPLUG/raw/main/assets/downstream_data.jpg)
#### 摘要
為了推動視覺語言預訓練（VLP）和多模態大語言模型（LLM）在中文社區的發展，我們首先發布了最大的公共中文高質量視頻語言數據集優酷-mPLUG，該數據集收集於優酷，中國知名視頻分享網站，有著嚴格的安全性、多樣性和質量標準。Youku-mPLUG 包含從 4 億個原始視頻中過濾出來的 1000 萬個中文視頻文本對，涵蓋 45 個不同類別，用於大規模預訓練。此外，為了促進對視頻語言模型的全面評估，我們精心構建了最大的人工註釋中文基準，涵蓋跨模態檢索、視頻字幕和視頻類別分類這三個流行的視頻語言任務。Youku-mPLUG可以讓研究人員在未來進行更深入的多模態研究並開發出更好的應用。此外，我們還發布了流行的視頻語言預訓練模型 ALPRO 和 mPLUG-2，以及我們提出的模塊化解碼器模型 mPLUG-在 Youku-mPLUG 上預訓練的視頻。

---------

### [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://paperswithcode.com/paper/mimic-it-multi-modal-in-context-instruction)[2023年6月8日]
![](https://camo.githubusercontent.com/70613ab882a7827808148a2c577029d544371e707b0832a0b01151c54ce553c3/68747470733a2f2f692e706f7374696d672e63632f5477315a304243572f6f7474657276302d322d64656d6f2e706e67)
[luodian/otter](https://github.com/luodian/otter)
#### 摘要
對於涉及復雜視覺場景的交互式視覺語言任務，需要大量多樣化和創造性的指令響應對來調整視覺語言模型（VLM）。然而，目前視覺-語言指令-響應對在數量、多樣性和創造力方面的可用性仍然有限，這對交互式 VLM 的推廣提出了挑戰。在這裡，我們提出了多模態上下文指令調優 (MIMIC-IT)，這是一個包含 280 萬個多模態指令響應對的數據集，其中**有 220 萬條**來自圖像和視頻的獨特指令。每對都伴隨著多模式上下文信息，形成旨在增強 VLM 感知、推理和規劃能力的對話環境。被稱為 Syphus 的指令響應收集過程使用自動註釋管道進行擴展，該管道將人類專業知識與 GPT 的功能相結合。使用 MIMIC-IT 數據集，我們訓練了一個名為 Otter 的大型 VLM。根據對視覺語言基准進行的廣泛評估，觀察到 Otter 在多模式感知、推理和情境學習方面表現出卓越的熟練程度。人工評估表明它有效地符合用戶的意圖。我們發布了 MIMIC-IT 數據集、指令響應收集管道、基準測試和 Otter 模型。

---------

# 開源模型
## LLM
### [PolyLM: An Open Source Polyglot Large Language Model](https://paperswithcode.com/paper/polylm-an-open-source-polyglot-large-language)[2023年7月12日]
![](https://hackmd.io/_uploads/rJlCkNCoh.png=80%x)
[modelscope/modelscope](https://github.com/modelscope/modelscope)
[Download](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation)
#### 摘要
大型語言模型 (LLM) 表現出卓越的理解、推理和生成以下自然語言指令的能力。然而，法學碩士的發展主要集中在英語等高資源語言上，從而限制了其在其他語言中的適用性和研究。因此，我們推出了 PolyLM，這是一種在 6400 億個 (B) Token 上進行訓練的多語言 LLM，有兩種模型大小：1.7B 和 13B。為了增強其多語言能力，我們1）將雙語數據集成到訓練數據中；2）採用課程學習策略，在預訓練時將非英語數據的比例從第一階段的30%增加到最後階段的60%。此外，我們提出了一種自動生成 132 的多語言自指令方法。7K多種多語言指令用於模型微調。

--------
### [h2oGPT: Democratizing Large Language Models](https://paperswithcode.com/paper/h2ogpt-democratizing-large-language-models)[2023年6月13日]
![](https://github.com/h2oai/h2ogpt/raw/main/docs/ui_talk_to_images.png =80%x)
[h2oai/h2ogpt](https://github.com/h2oai/h2ogpt)
#### 摘要
h2oGPT是一套開源代碼存儲庫，用於創建和使用基於生成預訓練 Transformer (GPT) 的 LLM。該項目的目標是創建世界上最好的真正開源替代閉源方法。作為令人難以置信且勢不可擋的開源社區的一部分，我們與令人難以置信的不可阻擋的開源社區合作，並開源了多個經過微調的 h2oGPT 模型，包含 7 至 400 億個參數，可在完全寬鬆的 Apache 2.0 許可證下用於商業用途。

--------

### [GLM-130B: An Open Bilingual Pre-trained Model](https://paperswithcode.com/paper/glm-130b-an-open-bilingual-pre-trained-model)
[thudm/glm-130b](https://github.com/thudm/glm-130b)

| Hardware | GPU Memory | Quantization |Weight Offload |
| -------- | -------- | -------- |-------- |
| 8 * A100 |   40 GB  | No | No |
| 8 * V100 |   32 GB  | No | Yes (BMInf) |
| 8 * V100 |   32 GB  | INT8 | No |
| 8 * RTX 3090 |   24 GB  | INT8 | No |
| 4 * RTX 3090 |   24 GB  | INT4 | No |
| 8 * RTX 2080 Ti|  11 GB  | INT4 | No |

#### 摘要
我們推出 GLM-130B，一個具有 1300 億個參數的雙語（英語和中文）預訓練語言模型。這是一種開源至少與 GPT-3 一樣好的 100B 規模模型的嘗試，並揭示如何成功地預訓練這種規模的模型。在這一努力的過程中，我們面臨著許多意想不到的技術和工程挑戰，特別是在損失峰值和不收斂方面。在本文中，我們介紹了 GLM-130B 的訓練過程，包括其設計選擇、效率和穩定性的訓練策略以及工程工作。由此產生的 GLM-130B 模型在各種流行的英語基準測試中提供了明顯優於 GPT-3 175B 的性能，而 OPT-175B 和 BLOOM-176B 中沒有觀察到性能優勢。它的性能也始終顯著優於 ERNIE TITAN 3。0 260B——最大的中文語言模型——跨越相關基準。最後，我們利用 GLM-130B 獨特的縮放屬性來實現 INT4 量化，無需量化感知訓練，幾乎沒有性能損失，使其成為 100B 縮放模型中的第一個。

--------

### [LLaMA: Open and Efficient Foundation Language Models](https://paperswithcode.com/paper/llama-open-and-efficient-foundation-language-1)[arXiv 2023]
![](https://s4.itho.me/sites/default/files/styles/picture_size_large/public/field/image/0227-llama-960.jpg?itok=k7-_4_mA)[facebookresearch/llama](https://github.com/facebookresearch/llama)

#### 摘要
我們介紹 LLaMA，這是一個基礎語言模型的集合，參數範圍從 7B 到 65B。我們在數万億個代幣上訓練我們的模型，並表明可以專門使用公開可用的數據集來訓練最先進的模型，而無需訴諸專有和無法訪問的數據集。特別是，LLaMA-13B 在大多數基準測試中都優於 GPT-3 (175B)，而 LLaMA-65B 可以與最好的模型 Chinchilla-70B 和 PaLM-540B 競爭。我們向研究界發布了所有模型。

--------

### [Panda LLM: Training Data and Evaluation for Open-Sourced Chinese Instruction-Following Large Language Models](https://paperswithcode.com/paper/panda-llm-training-data-and-evaluation-for)[2023年5月4日]
![](https://github.com/dandelionsllm/pandallm/raw/main/panda_logo.PNG =40%x)
[dandelionsllm/pandallm](https://github.com/dandelionsllm/pandallm)
#### 摘要
該項目的重點是通過指令調整來增強開源大型語言模型並對其性能進行全面評估。我們探討了各種訓練數據因素（例如數量、質量和語言分佈）如何影響在可公開訪問的英語和中文高質量教學數據集上訓練的指令調整模型的性能。我們的目標是通過定量分析來補充評估，為開源聊天模型的持續發展提供有價值的見解。我們的模型、數據和代碼是公開的，可供其他人使用和構建。

--------

### [Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data ](https://paperswithcode.com/paper/baize-an-open-source-chat-model-with)[2023年4月3日]
![](https://user-images.githubusercontent.com/22514219/229863275-0e83c1cf-0661-4afa-9a47-1ce20fb521ae.gif)
[project-baize/baize-chatbot](https://github.com/project-baize/baize-chatbot)
#### 摘要
ChatGPT 等聊天模型已經顯示出令人印象深刻的功能，並已在眾多領域迅速採用。然而，這些模型只能通過受限的 API 訪問，這為該領域的新研究和進展造成了障礙。我們提出了一種管道，可以利用 ChatGPT 與自身進行對話，自動生成高質量的多輪聊天語料庫。隨後，我們採用參數高效的調優來增強 LLaMA（一種開源大型語言模型）。由此產生的模型名為Baize，在帶有護欄的多輪對話中表現出了良好的性能，可以最大限度地減少潛在風險。此外，提出了一種名為**帶反饋的自蒸餾**的新技術，以利用 ChatGPT 的反饋進一步提高 Baize 模型的性能。

-----------

### [FinGPT: Open-Source Financial Large Language Models](https://paperswithcode.com/paper/fingpt-open-source-financial-large-language)[2023年6月9日]
![](https://github.com/AI4Finance-Foundation/FinGPT/raw/master/figs/FinGPT_framework.png)
[AI4Finance-Foundation/FinGPT](https://github.com/ai4finance-foundation/fingpt)
[AI4Finance-Foundation/FinNLP](https://github.com/ai4finance-foundation/finnlp)
#### 摘要
大型語言模型（LLM）已經顯示出在不同領域徹底改變自然語言處理任務的潛力，引發了人們對金融的極大興趣。獲取高質量的金融數據是金融法學碩士（FinLLM）面臨的首要挑戰。雖然像 BloombergGPT 這樣的專有模型已經利用了其獨特的數據積累，但這種特權訪問需要一種開源替代方案來實現互聯網規模金融數據的民主化。在本文中，我們提出了一種針對金融領域的開源大型語言模型 FinGPT。與專有模型不同，FinGPT 採用以數據為中心的方法，為研究人員和從業人員提供可訪問且透明的資源來開發他們的 FinLLM。我們強調自動數據管理管道和輕量級低秩適應技術在構建 FinGPT 中的重要性。此外，我們還展示了幾種潛在的應用程序作為用戶的墊腳石，例如機器人諮詢、算法交易和低代碼開發。

------------

### [ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge](https://paperswithcode.com/paper/chatdoctor-a-medical-chat-model-fine-tuned-on)[2023年3月24日]
![](https://github.com/Kent0n-Li/ChatDoctor/raw/main/fig/overview.PNG)
[Kent0n-Li/ChatDoctor](https://github.com/kent0n-li/chatdoctor)
#### 摘要
這項研究的主要目的是通過創建一種專門的語言模型來提高醫療建議的準確性，從而解決 ChatGPT 等流行的大型語言模型 (LLM) 的醫學知識中觀察到的局限性。 我們通過使用來自廣泛使用的在線醫療諮詢平台的 100,000 條醫患對話的大型數據集來調整和完善大型語言模型元人工智能 (LLaMA)，從而實現了這一目標。 為了尊重隱私問題，這些對話都經過清理和匿名處理。 除了模型細化之外，我們還採用了自我導向的信息檢索機制，允許模型訪問和利用來自維基百科等在線資源的實時信息以及來自精選離線醫學數據庫的數據。 通過現實世界的醫患互動對模型進行微調，顯著提高了模型了解患者需求並提供明智建議的能力。 通過為模型配備來自可靠的在線和離線來源的自主信息檢索，我們觀察到其響應的準確性有了顯著提高。 我們提出的 ChatDoctor 代表了醫學法學碩士的重大進步，展示了在理解患者詢問和提供準確建議方面的顯著進步。 鑑於醫療領域的高風險和低容錯性，這種提供準確可靠信息的增強不僅是有益的，而且是必要的。

---------

### [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://paperswithcode.com/paper/huatuo-tuning-llama-model-with-chinese)[2023年4月14日]
![](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/raw/main/assets/case.png)
[SCIR-HI/Huatuo-Llama-Med-Chinese](https://github.com/scir-hi/huatuo-llama-med-chinese)
#### 摘要
大型語言模型 (LLM)，例如 LLaMA 模型，已在各種通用領域自然語言處理 (NLP) 任務中證明了其有效性。然而，由於響應中需要醫學專業知識，法學碩士在生物醫學領域任務中尚未表現最佳。為了應對這一挑戰，我們提出了 HuaTuo，這是一種基於 LLaMA 的模型，已使用生成的 QA（問題-答案）實例進行了監督微調。實驗結果表明，華佗產生的反應具有更可靠的醫學知識。

--------

# Benchmark
## Multimodal
### [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://paperswithcode.com/paper/mme-a-comprehensive-evaluation-benchmark-for)[2023年6月23日]
![](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/raw/main/images/xmind.png)
[bradyfu/awesome-multimodal-large-language-models](https://github.com/bradyfu/awesome-multimodal-large-language-models)
#### 摘要
多模態大語言模型（MLLM）依靠強大的LLM來執行多模態任務，在最近的研究中顯示出驚人的新興能力，例如基於圖像寫詩。然而，這些案例研究很難全面反映MLLM的績效，缺乏全面的評估。在本文中，我們填補了這一空白，提出了第一個 MLLM 評估基準 MME。它測量總共 14 個子任務的感知和認知能力。為了避免直接使用公共數據集進行評估可能導致的數據洩露，指令-答案對的註釋都是手動設計的。簡潔的指令設計使我們能夠公平地比較 MLLM，而不是在即時工程中苦苦掙扎。而且，有了這樣的指示，我們還可以輕鬆地進行定量統計。在我們的 MME 上對總共 12 個先進的 MLLM 進行了全面評估，這不僅表明現有的 MLLM 仍然有很大的改進空間，而且還揭示了後續模型優化的潛在方向。


--------

# 科学空间|Scientific Spaces議題
如果有人想要學習關於ML當中的數學、物理推導等東西，極力推薦「**苏剑林|BoJon**」的[「**科学空间|Scientific Spaces」**](https://kexue.fm/)。
個人推薦的議題請到另一篇文章[**科学空间|Scientific Spaces議題**](https://hackmd.io/sEVKyc0oTPKkrjbPURI4FA)查看。


--------


下面的Paper有夠偏門，都是臭打遊戲的

# RL(Reinforcement Learning)
## 玩遊戲
### [alex-petrenko/sample-factory](https://paperswithcode.com/paper/counter-strike-deathmatch-with-large-scale)[2021年4月9日]
![](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/raw/main/gif_data_01.gif)

![](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/raw/main/NN_overview.png =70%x)
[TeaPearce/Counter-Strike_Behavioural_Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning#trained-models)
#### 摘要
本文描述了一個可以玩流行的第一人稱射擊 (FPS) 視頻遊戲《反恐精英》的 AI 代理。來自像素輸入的全球攻勢（CSGO）。該代理是一個深度神經網絡，與死亡競賽遊戲模式中中等難度的內置人工智能的性能相匹配，同時採用了類人的遊戲風格。與之前的遊戲工作不同，CSGO 沒有可用的 API，因此算法必須實時訓練和運行。這限制了可以生成的策略數據的數量，從而排除了許多強化學習算法。我們的解決方案使用行為克隆——對在線服務器上從人類游戲中抓取的大型噪聲數據集（400 萬幀，大小與 ImageNet 相當）和高質量專家演示的較小數據集進行訓練。

--------

### [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning](https://paperswithcode.com/paper/sample-factory-egocentric-3d-control-from)[ICML2020]
![](https://github.com/alex-petrenko/sf_assets/raw/main/gifs/vizdoom.gif?raw=true)
![](https://github.com/alex-petrenko/sf_assets/raw/main/gifs/isaac.gif?raw=true)
![](https://github.com/alex-petrenko/sf_assets/raw/main/gifs/mujoco.gif?raw=true)
#### 摘要
強化學習實驗規模的擴大使研究人員能夠在視頻遊戲的複雜智能體訓練以及機器人的模擬到真實遷移方面取得前所未有的成果。通常，此類實驗依賴於大型分佈式系統，並且需要昂貴的硬件設置，從而限制了對這一令人興奮的研究領域的更廣泛的訪問。在這項工作中，我們的目標是通過優化強化學習算法的效率和資源利用率而不是依賴分佈式計算來解決這個問題。我們推出了“樣本工廠”，這是一個針對單機設置優化的高吞吐量培訓系統。我們的架構將高效、異步、基於 GPU 的採樣器與離策略校正技術相結合，使我們能夠實現高於$10^5$環境幀/秒處理 3D 中的重要控制問題，而不犧牲樣本效率。我們擴展了 Sample Factory 以支持自我遊戲和基於群體的訓練，並應用這些技術來訓練用於多人第一人稱射擊遊戲的高能力智能體。