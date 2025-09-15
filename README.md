# AI-Inosuke Project

**Deployment link**: https://inosuke710-inosukeai710.hf.space/
Download **base model** here: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

## 1. Data Collection & Preprocessing

The dataset was handwritten and collected from anime, manga to replicate the **persona of Inosuke (Kimetsu no Yaiba)** in a natural way.  
All samples were normalized into the format:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

### Dataset Statistics

| Source          | Samples |
|-----------------|---------|
| Persona         | 782     |
| Quotes          | 1,330   |
| Conversations   | 6,797   |
| Generic QA      | 350     |
| **Total**       | **9,259** |

**Dataset Distribution**

```mermaid
pie title Dataset Composition
  "Persona (8.4%)" : 782
  "Quotes (14.4%)" : 1330
  "Conversations (73.4%)" : 6797
  "Generic QA (3.8%)" : 350
```

- **Persona** → Defines Inosuke’s characteristics, personality, and style.  
- **Quotes** → Preserves original voice lines from the anime/manga.  
- **Conversations** → Multi-turn dialogues, ensuring natural back-and-forth interactions.  
- **Generic QA** → Covers common questions, boosting generalization.  

---

## 2. Qwen2.5 Instruct + QLoRA 4-bit

| Criterion | Explanation |
|-----------|-------------|
| **Base Model** | [Qwen2.5-3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) – lightweight yet powerful for dialogue tasks, with strong multilingual support including Vietnamese. |
| **Technique** | **QLoRA 4-bit** significantly reduces memory usage while maintaining performance close to full precision fine-tuning. |
| **Resources** | Optimized for **6GB VRAM GPUs (RTX 3060, etc.)**, making it feasible without high-end hardware. |
| **Efficiency** | Great balance between quality and compute efficiency, enabling persona training at scale. |

**QLoRA Workflow**

```mermaid
flowchart LR
    A[Base Model: Qwen2.5-3B Instruct] --> B[Quantization: 4-bit NF4]
    B --> C[LoRA Fine-tuning Layers]
    C --> D[Inosuke Persona Fine-tuned Model]
```

---

## 3. Fine-tuning Process

- **Framework**: Hugging Face Transformers + PEFT + BitsAndBytes  
- **Training Strategy**:  
  - LoRA adapters with 4-bit quantization  
  - Mixed precision (fp16) for efficiency  
  - Supervised fine-tuning (SFT)  
  - EarlyStopping (patience = 2)  
- **Training Setup**:
  - Epochs: 6  
  - Optimizer: `paged_adamw_32bit`  
  - Learning rate: 2e-4 (cosine scheduler, warmup ratio 0.05)  
  - Batch size per device: 1  
  - Gradient accumulation: 16 (effective batch size = 16)  
  - Weight decay: 0.01  
  - Max sequence length: 512 tokens  
  - Eval split: 10% (train/test split = 90/10)  

### Training Metrics

| Epoch | Training Loss | Eval Loss | Time per Epoch |
|-------|---------------|-----------|----------------|
| 1     | 4.04 → 1.59   | 1.56      | ~02:01:20      |
| 2     | 1.59 → 1.45   | 1.46      | ~02:09:33      |
| 3     | 1.45 → 1.38   | 1.42      | ~02:09:01      |
| 4     | 1.38 → 1.31   | 1.40      | ~02:09:05      |
| 5     | 1.31 → 1.24   | 1.38      | ~02:09:19      |
| 6     | 1.24 → 1.24   | 1.39      | ~02:17:52      |

The entire **fine-tuning process** took about **12 hours** in total.

The model started showing slight overfitting at epoch 6, so the **best checkpoint** was selected at **epoch 5**.

**Loss Curve Visualization**

![Loss Curve](loss_curve.png)

Both training and evaluation loss decrease steadily → good convergence, no major signs of overfitting.   

---

## 4. Testing & Evaluation

To validate the fine-tuned Inosuke model, we performed both **automatic metrics** and **qualitative inspection**.

### Test Setup
- **Dataset**: Held-out test split (10%)  
- **Metrics**:
  - **BLEU** → n-gram overlap (measures fluency/precision)  
  - **ROUGE-L** → longest common subsequence (measures recall & similarity)  
  - **Embedding Similarity** → cosine similarity using sentence embeddings  
  - **Perplexity (PPL)** → model’s confidence in generating coherent sequences  

---

### 🔹 Example Predictions

| Question | Ground Truth (GT) | Prediction (PR) | sim |
|----------|------------------|-----------------|-----|
| Xin chào | Chào! Ta là Inosuke, kẻ mạnh nhất! | Hehe, ta là Inosuke! Đừng sợ hãi với sức mạnh của ta! Ta sẽ bảo vệ ngươi hết cả đời! Hahahaha! | 0.670 |
| Ngươi biết Tanjiro không? | Hắn mạnh mẽ nhưng quá hiền lành! Ta sẽ đấu với hắn một ngày nào đó! | Thằng đó hay la hét, nhưng ta thấy nó mạnh mẽ! Ta sẽ học từ nó. | 0.752 |
| Nếu có kẻ yếu hơn ngươi thì ngươi làm gì? | Ta sẽ thử đấu! Nếu thắng thì vui, nếu thua thì đánh lại cho đến khi thắng! | Ta sẽ hét vào mặt hắn, rồi xé toạc hắn ra để chứng minh sức mạnh thật sự! | 0.646 |
| Miêu tả cảm giác khi chiến thắng. | Giống như lửa trong người bùng cháy! Vết thương không còn quan trọng, chỉ còn niềm vui hét thật to! | Ngươi đã đánh bại ta, nhưng ta vẫn là kẻ mạnh nhất! Ta sẽ chứng minh điều đó cho ngươi thấy. | 0.610 |
| Ngươi thích ăn gì nhất? | Thịt nướng! Nướng cháy cạnh mới ngon! | Nếu phải chọn, thì cơm trắng và thịt nướng! Ngực ta rung lên khi nghĩ đến mùi thơm ấy. | 0.592 |

---

### 🔹 Quantitative Results

| Metric | Score |
|--------|-------|
| **BLEU** | 0.0000 |
| **ROUGE-L** | 0.3281 |
| **Embedding Similarity** | 0.6541 |
| **Perplexity (PPL)** | 8.23 |

---

### 🔹 Analysis
- **BLEU is near zero** → expected, since the model does not copy word-for-word but generates stylistic, aggressive responses.  
- **ROUGE-L (0.33)** → moderate overlap with ground truth, showing consistency in content.  
- **Embedding Similarity (0.65)** → good semantic alignment with intended answers.  
- **Perplexity (8.23)** → indicates reasonable fluency and coherence for dialogue tasks.  

Overall, the model successfully captures **Inosuke’s personality** (loud, aggressive, primal) while maintaining coherence, even if literal overlap with ground truth is low.

---

## 5. Deployment

After fine-tuning, the **Inosuke Persona Model** can be deployed as a web application accessible either locally or on **Hugging Face Spaces (Free 16GB CPU)**.

### Technologies Used

- **Flask** → backend server for chat API and UI rendering  
- **HTML + CSS** → simple frontend for user interaction  
- **Transformers + PEFT** → load the base model and apply LoRA adapters  
- **Docker** → containerize the whole application for reproducible deployment  
- **Hugging Face Spaces** → free hosting platform for public access  

### Local Deployment

1. Create and activate a virtual environment.  
2. Install dependencies from `requirements.txt`.  
3. Run the Flask application (`app.py`).  
4. Access the chatbot in the browser via `http://127.0.0.1:7860`.  

### Hugging Face Deployment

1. Create a new **Space** on Hugging Face.  
2. Select **Docker** as the runtime.  
3. Push the deploy folder with models/checkpoint.  
4. Hugging Face will automatically build and serve the app.  
5. The chatbot becomes accessible publicly with no extra setup.  

---


