import os
import time
import streamlit as st
from PIL import Image
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from gradio_client import Client as GradioClient, handle_file

# Set Streamlit page config
st.set_page_config(
    page_title="AI Fashion & Virtual Try-On Assistant",
    page_icon="👗",
    layout="wide"
)

# Load environment variables
load_dotenv()

st.title("👗 AI Fashion Assistant & Virtual Try-On")
st.markdown(
    "Demonstration platform for **Multimodal Product Search (CLIP + Pinecone)** and **Generative Virtual Try-On (IDM-VTON)**."
)

# Sidebar - System & Hardware status
st.sidebar.header("⚙️ System Status")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Compute Device: **{device.upper()}** (GPU Accelerated)" if device == "cuda" else "Compute Device: CPU")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_2XvhS_F8DBNuXvffNAUVDRnYUccr21mwaPQ7Xwpa7wJLW8Ztgp8a2jpGeQKVMFJ4THENJ")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-image-search")

# Cached resource loading
@st.cache_resource(show_spinner="Loading CLIP model & Pinecone index...")
def init_search_resources():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    images = ds["image"]
    return index, clip_model, images

@st.cache_resource(show_spinner="Connecting to Virtual Try-On API (Hugging Face)...")
def init_tryon_client():
    return GradioClient("Nymbo/Virtual-Try-On", token=os.getenv("HF_TOKEN"))

# Initialize models
try:
    index, clip_model, dataset_images = init_search_resources()
    st.sidebar.success(f"Pinecone Index: **{INDEX_NAME}** (44,000+ items)")
except Exception as e:
    st.sidebar.error(f"Error loading search resources: {e}")
    index, clip_model, dataset_images = None, None, None

tab1, tab2, tab3, tab4 = st.tabs([
    "✨ 1. Virtual Try-On",
    "🔍 2. Text-Based Search",
    "🖼️ 3. Image-Based Search",
    "📱 4. WhatsApp Bot Setup"
])

# ----------------- TAB 1: VIRTUAL TRY-ON -----------------
with tab1:
    st.header("AI-Powered Virtual Try-On (IDM-VTON)")
    st.write("Upload a photo of a person and a garment image to see the garment fitted on the person.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Person Image")
        use_sample_person = st.checkbox("Use sample person image", value=True)
        if use_sample_person and os.path.exists("data/sample_tryon/person.jpg"):
            person_img_path = "data/sample_tryon/person.jpg"
            st.image(person_img_path, caption="Sample Person", width=250)
        else:
            uploaded_person = st.file_uploader("Upload Person Photo", type=["jpg", "jpeg", "png"], key="person")
            if uploaded_person:
                person_img_path = "temp_person.jpg"
                with open(person_img_path, "wb") as f:
                    f.write(uploaded_person.getbuffer())
                st.image(person_img_path, caption="Uploaded Person", width=250)
            else:
                person_img_path = None

    with col2:
        st.subheader("2. Garment Image")
        use_sample_cloth = st.checkbox("Use sample garment image", value=True)
        if use_sample_cloth and os.path.exists("data/sample_tryon/garment.jpg"):
            cloth_img_path = "data/sample_tryon/garment.jpg"
            st.image(cloth_img_path, caption="Sample Garment", width=250)
        else:
            uploaded_cloth = st.file_uploader("Upload Garment Photo", type=["jpg", "jpeg", "png"], key="cloth")
            if uploaded_cloth:
                cloth_img_path = "temp_cloth.jpg"
                with open(cloth_img_path, "wb") as f:
                    f.write(uploaded_cloth.getbuffer())
                st.image(cloth_img_path, caption="Uploaded Garment", width=250)
            else:
                cloth_img_path = None

    garment_desc = st.text_input("Garment Description (Optional)", value="A stylish topwear garment")

    if st.button("🚀 Run Virtual Try-On", type="primary"):
        if not person_img_path or not cloth_img_path:
            st.warning("Please provide both a person photo and a garment photo.")
        else:
            with st.spinner("Generating try-on with IDM-VTON diffusion model (typically takes 25-40 seconds)..."):
                try:
                    gradio_client = init_tryon_client()
                    start_time = time.time()
                    result = gradio_client.predict(
                        dict={"background": handle_file(person_img_path), "layers": [], "composite": None},
                        garm_img=handle_file(cloth_img_path),
                        garment_des=garment_desc,
                        is_checked=True,
                        is_checked_crop=False,
                        denoise_steps=25,
                        seed=42,
                        api_name="/tryon"
                    )
                    elapsed = time.time() - start_time
                    if result and len(result) > 0:
                        output_image_path = result[0]
                        st.success(f"Try-On synthesis completed in {elapsed:.1f} seconds! 🎉")
                        st.subheader("Generated Result")
                        res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])
                        with res_col1:
                            st.image(person_img_path, caption="Original Person", use_container_width=True)
                        with res_col2:
                            st.image(cloth_img_path, caption="Selected Garment", use_container_width=True)
                        with res_col3:
                            st.image(output_image_path, caption="✨ AI Virtual Try-On Result", use_container_width=True)
                    else:
                        st.error("The API did not return an output image. Please try again.")
                except Exception as ex:
                    st.error(f"Try-On Error: {ex}")

# ----------------- TAB 2: TEXT SEARCH -----------------
with tab2:
    st.header("Natural Language Product Search (CLIP + Pinecone)")
    st.write("CLIP translates your text into a 512-dim embedding and finds the nearest products in the 44,000+ catalog.")

    search_query = st.text_input("Enter fashion search query:", value="French Connection blue jeans for men")
    top_k = st.slider("Number of results to display", min_value=1, max_value=8, value=4, key="text_top_k")

    if st.button("🔍 Search Catalog", key="text_search_btn"):
        if not search_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner(f"Encoding '{search_query}' and querying Pinecone..."):
                query_vec = clip_model.encode(search_query).tolist()
                response = index.query(top_k=top_k, vector=query_vec, include_metadata=True)
                matches = response.get("matches", [])
                
                if matches:
                    st.write(f"Found {len(matches)} matching products:")
                    cols = st.columns(len(matches))
                    for i, match in enumerate(matches):
                        img_idx = int(match["id"])
                        prod_img = dataset_images[img_idx]
                        meta = match.get("metadata", {})
                        with cols[i]:
                            st.image(prod_img, caption=meta.get("productDisplayName", f"Item #{img_idx}"), use_container_width=True)
                            st.caption(f"**Category:** {meta.get('articleType', 'N/A')}\n\n**Color:** {meta.get('baseColour', 'N/A')}\n\n**Score:** {match['score']:.2f}")
                else:
                    st.warning("No matches found.")

# ----------------- TAB 3: IMAGE SEARCH -----------------
with tab3:
    st.header("Visual Similarity Product Search")
    st.write("Upload an image of clothing to discover visually identical or similar items in the catalog.")

    img_col1, img_col2 = st.columns([1, 2])
    with img_col1:
        use_sample_search_img = st.checkbox("Use sample search image", value=True, key="search_sample")
        if use_sample_search_img and os.path.exists("data/sample_tryon/garment.jpg"):
            search_img_path = "data/sample_tryon/garment.jpg"
            st.image(search_img_path, caption="Query Image", width=220)
        else:
            uploaded_search_img = st.file_uploader("Upload clothing image", type=["jpg", "png", "jpeg"], key="img_search")
            if uploaded_search_img:
                search_img_path = "temp_query_img.jpg"
                with open(search_img_path, "wb") as f:
                    f.write(uploaded_search_img.getbuffer())
                st.image(search_img_path, caption="Uploaded Query Image", width=220)
            else:
                search_img_path = None

    with img_col2:
        img_top_k = st.slider("Number of results to display", min_value=1, max_value=8, value=4, key="img_top_k")
        if st.button("🔎 Find Visually Similar Products", key="img_search_btn"):
            if not search_img_path:
                st.warning("Please provide a query image.")
            else:
                with st.spinner("Extracting visual feature embeddings with CLIP..."):
                    query_img = Image.open(search_img_path).convert("RGB").resize((224, 224))
                    dense_vec = clip_model.encode(query_img, convert_to_tensor=True).cpu().tolist()
                    response = index.query(top_k=img_top_k, vector=dense_vec, include_metadata=True)
                    matches = response.get("matches", [])

                    if matches:
                        cols = st.columns(len(matches))
                        for i, match in enumerate(matches):
                            img_idx = int(match["id"])
                            prod_img = dataset_images[img_idx]
                            meta = match.get("metadata", {})
                            with cols[i]:
                                st.image(prod_img, caption=meta.get("productDisplayName", f"Item #{img_idx}"), use_container_width=True)
                                st.caption(f"**Category:** {meta.get('articleType', 'N/A')}\n\n**Color:** {meta.get('baseColour', 'N/A')}\n\n**Score:** {match['score']:.2f}")

# ----------------- TAB 4: WHATSAPP BOT SETUP -----------------
with tab4:
    st.header("📱 Running via WhatsApp (Twilio + Flask + Ngrok)")
    st.markdown("""
    To run this whole system over WhatsApp:
    
    1. **Fill in your Twilio Credentials in `.env`:**
       ```env
       TWILIO_ACCOUNT_SID=your_account_sid_here
       TWILIO_AUTH_TOKEN=your_auth_token_here
       TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
       ```
    2. **Start the Flask webhook server:**
       ```powershell
       .\\.venv\\Scripts\\python.exe src/app.py
       ```
       *(Runs on `http://127.0.0.1:8080`)*
       
    3. **Start Ngrok to expose port 8080 to the internet:**
       ```powershell
       ngrok http 8080
       ```
    4. **Configure Twilio Sandbox Webhook:**
       - Go to your Twilio Console -> Messaging -> Try it Out -> WhatsApp Sandbox.
       - Set the **WHEN A MESSAGE COMES IN** webhook URL to:
         `https://<your-ngrok-subdomain>.ngrok-free.app/webhook` (HTTP POST)
       - Update `NGROK_URL=https://<your-ngrok-subdomain>.ngrok-free.app` in your `.env`.
       
    5. **Test on WhatsApp:**
       - Send `0` for Virtual Try-On
       - Send `1` for Text Search
       - Send `2` for Image Search
    """)

