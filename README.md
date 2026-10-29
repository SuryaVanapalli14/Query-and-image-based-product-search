# 👗 Multimodal E-Commerce Assistant: Vector Product Search & Virtual Try-On with WhatsApp Integration

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%20CUDA-orange.svg)](https://pytorch.org/)
[![Vector DB](https://img.shields.io/badge/Pinecone-Serverless-green.svg)](https://www.pinecone.io/)
[![Model](https://img.shields.io/badge/CLIP-ViT--B--32-purple.svg)](https://huggingface.co/sentence-transformers/clip-ViT-B-32)
[![VTON](https://img.shields.io/badge/IDM--VTON-Diffusion-red.svg)](https://huggingface.co/spaces/Nymbo/Virtual-Try-On)
[![UI](https://img.shields.io/badge/Streamlit-Web%20Dashboard-red.svg)](https://streamlit.io/)
[![API](https://img.shields.io/badge/Twilio-WhatsApp%20API-teal.svg)](https://www.twilio.com/)

An end-to-end, AI-powered e-commerce shopping assistant combining **Multimodal Vector Search (OpenAI CLIP + Pinecone Serverless)**, **Generative Virtual Try-On (IDM-VTON Diffusion)**, and **Conversational Commerce (WhatsApp & Web)** to eliminate product discovery friction and reduce sizing-related return rates.

---

## 📌 Problem Statement & Business Value

Online apparel commerce faces two fundamental challenges:
1. **Catalog Search Friction**: Traditional keyword search (SQL `LIKE` or basic text match) fails when customers express queries naturally (e.g., *"aesthetic blue party wear"*) or search using a photo of an outfit they saw on social media.
2. **High E-Commerce Return Rates**: Fashion return rates hover between **30% to 40%**, primarily driven by fit uncertainty and difficulty visualizing how garments look on real individuals.

### **Our Solution: Hybrid Edge-Cloud AI Platform**
* **Cross-Modal Catalog Search**: Translates both natural language text queries and product photos into a shared 512-dimensional vector space via **CLIP ViT-B-32**, querying 44,000+ fashion catalog products in milliseconds.
* **Realistic Virtual Try-On**: Generates realistic garment-on-person composite images using **IDM-VTON (Image-based Virtual Try-on Network)** via high-performance cloud GPUs.
* **Accessible Omnichannel Interface**: Delivers the experience directly inside **WhatsApp** (the world's most widely used messaging app) alongside an interactive **Streamlit Web Demo**.

---

## 🏛️ System Architecture

```
                  +----------------------------------------------+
                  |                 USER CLIENTS                 |
                  |     (WhatsApp Messenger  /  Web Browser)     |
                  +----------------------+-----------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      [WhatsApp Messages]                                 [Interactive Web UI]
               v                                                   v
   +-----------------------+                             +--------------------+
   |   Twilio Webhook API   |                             |   Streamlit App    |
   |      (via Ngrok)      |                             |   (demo_web.py)    |
   +-----------+-----------+                             +---------+----------+
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Flask Webhook Controller    |
                         |          (src/app.py)         |
                         +---------------+---------------+
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
         [Search Pipeline]                               [Try-On Pipeline]
                 v                                               v
   +---------------------------+                   +---------------------------+
   |   OpenAI CLIP ViT-B-32    |                   |   Nymbo/Virtual-Try-On    |
   |  (Local GPU Acceleration) |                   |  (Hugging Face Cloud GPU) |
   |  Text & Image Embeddings  |                   |    IDM-VTON Diffusion     |
   +-------------+-------------+                   +-------------+-------------+
                 |                                               |
                 v                                               |
   +---------------------------+                                 |
   |    Pinecone Vector DB     |                                 |
   |  44,000+ Catalog Embeds   |                                 |
   |  Serverless ANN Search    |                                 |
   +-------------+-------------+                                 |
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
                         +-------------------------------+
                         |  Generated Visual Results &   |
                         |       Catalog Matches         |
                         +-------------------------------+
```

---

## 🚀 Key Features

| Feature | Technology | Description |
| :--- | :--- | :--- |
| **Natural Language Search** | CLIP (`clip-ViT-B-32`) + Pinecone | Search through 44,000+ clothing products with semantic descriptions (e.g., *"dark blue french connection jeans"*). |
| **Visual Similarity Search** | CLIP Vision Encoder + Pinecone | Upload any picture of a garment; the system extracts visual vectors and retrieves visually identical or similar items. |
| **AI Virtual Try-On** | IDM-VTON Diffusion Pipeline | Takes a person photo and a target garment photo, realistically warping and fitting the clothes to the body posture in ~30 seconds. |
| **Interactive Web UI** | Streamlit | Complete browser dashboard to test Try-On, text search, and visual search with live GPU hardware monitoring. |
| **WhatsApp Chatbot** | Twilio API + Flask | Fully automated conversational bot allowing users to search and try garments directly through WhatsApp chat. |

---

## 📂 Project Structure

```
.
├── data/
│   ├── sample_tryon/           # Pre-loaded high-res sample person & garment images
│   │   ├── person.jpg
│   │   └── garment.jpg
│   └── download (1-5).png      # Catalog sample thumbnails
├── notebooks/
│   └── AI_Assistant_ecommerce.ipynb   # Exploratory notebook for indexing & embeddings
├── results_img/                # Architecture diagrams and sample demonstration images
├── src/
│   ├── app.py                  # Twilio WhatsApp webhook Flask server
│   └── hybridsearch.py         # Standalone CLIP + Pinecone vector query engine
├── .env                        # Local environment credentials (excluded from git)
├── .env.example                # Configuration template
├── .gitignore                  # Git tracking exclusion list
├── demo_web.py                 # Streamlit interactive web application
├── launch_web_demo.bat         # 1-click Windows launcher for Web Demo
├── launch_whatsapp_bot.bat     # 1-click Windows launcher for WhatsApp server
├── PROJECT_DOCUMENTATION.md    # In-depth technical documentation & viva guide
├── requirements.txt            # Python dependencies
└── test_cli.py                 # Command-line verification script
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA drivers (optional but recommended for fast local CLIP embedding)
* Active Internet connection (for Pinecone and Hugging Face APIs)

### 2. Environment Setup
Clone the repository and install dependencies:
```powershell
git clone https://github.com/SuryaVanapalli14/Query-and-image-based-product-search.git
cd Query-and-image-based-product-search

# Create and activate virtual environment
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Fill in your credentials:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=hybrid-image-search
HF_TOKEN=your_huggingface_access_token
```

---

## 🖥️ Running the Project

### Option A: Interactive Web Demo (Fastest & Recommended)
Double-click `launch_web_demo.bat` **OR** run:
```powershell
.\.venv\Scripts\python.exe -m streamlit run demo_web.py
```
Open **`http://localhost:8501`** in your browser.
* **Tab 1:** Run Virtual Try-On using pre-loaded or uploaded images.
* **Tab 2:** Test semantic text-based product search.
* **Tab 3:** Test image-based visual similarity search.

---

### Option B: Terminal / CLI Verification
Run the automated test suite to verify vector search and try-on:
```powershell
.\.venv\Scripts\python.exe test_cli.py
```

---

### Option C: WhatsApp Chatbot Integration

1. **Add Twilio Credentials to `.env`:**
   ```env
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   ```

2. **Launch the Flask Server:**
   Double-click `launch_whatsapp_bot.bat` **OR** run:
   ```powershell
   .\.venv\Scripts\python.exe src/app.py
   ```
   *(Server starts on `http://127.0.0.1:8080`)*

3. **Expose with Ngrok:**
   In a separate terminal:
   ```powershell
   ngrok http 8080
   ```
   Copy the generated forwarding URL (e.g., `https://xxxx.ngrok-free.app`) and set `NGROK_URL` in `.env`.

4. **Connect Twilio WhatsApp Sandbox:**
   * In the Twilio Console, go to **Messaging ➔ Try it Out ➔ Send a WhatsApp Message**.
   * Under **Sandbox Settings**, paste `https://xxxx.ngrok-free.app/webhook` as the **HTTP POST** URL.
   * Send `join <your-sandbox-word>` on WhatsApp to activate the bot.

5. **Interact via WhatsApp:**
   * Send `0`: Virtual Try-On (send person photo, then garment photo)
   * Send `1`: Text Search (send query e.g. "red floral dress")
   * Send `2`: Image Search (send photo of clothing)

---

## 🛡️ License & Acknowledgements
* **CLIP ViT-B-32**: OpenAI & Sentence-Transformers
* **IDM-VTON**: Nymbo / Yisol Image-based Virtual Try-On Network
* **Dataset**: `ashraq/fashion-product-images-small` (44,000+ fashion catalog images)
* **Vector Engine**: Pinecone Serverless
