"""
Quick CLI verification script for all 3 project features:
1. Natural Language Product Search (CLIP + Pinecone)
2. Image-Based Product Search (CLIP + Pinecone)
3. Virtual Try-On API (IDM-VTON via Hugging Face)
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

def test_search():
    print("=" * 60)
    print("1. TESTING TEXT & IMAGE PRODUCT SEARCH (CLIP + PINECONE)")
    print("=" * 60)
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    from PIL import Image

    api_key = os.getenv("PINECONE_API_KEY", "pcsk_2XvhS_F8DBNuXvffNAUVDRnYUccr21mwaPQ7Xwpa7wJLW8Ztgp8a2jpGeQKVMFJ4THENJ")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("hybrid-image-search")
    
    print("[*] Connecting to Pinecone index 'hybrid-image-search'...")
    stats = index.describe_index_stats()
    print(f"    Total items in catalog: {stats.total_vector_count}")

    print("[*] Loading CLIP model (SentenceTransformer clip-ViT-B-32)...")
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device='cuda')

    print("[*] Loading fashion dataset images...")
    ds = load_dataset('ashraq/fashion-product-images-small', split='train')
    images = ds['image']

    # 1. Text Search
    query = "casual blue jeans for men"
    print(f"\n[Text Query] Searching for: '{query}'")
    vec = model.encode(query).tolist()
    res = index.query(top_k=2, vector=vec, include_metadata=True)
    for i, match in enumerate(res['matches']):
        meta = match.get('metadata', {})
        print(f"  Match {i+1}: {meta.get('productDisplayName')} (Score: {match['score']:.2f})")

    # 2. Image Search
    if os.path.exists("test_cloth.jpg"):
        print("\n[Image Query] Searching with 'test_cloth.jpg'...")
        input_img = Image.open('test_cloth.jpg').convert('RGB').resize((224, 224))
        img_vec = model.encode(input_img, convert_to_tensor=True).cpu().tolist()
        img_res = index.query(top_k=2, vector=img_vec, include_metadata=True)
        for i, match in enumerate(img_res['matches']):
            meta = match.get('metadata', {})
            print(f"  Visual Match {i+1}: {meta.get('productDisplayName')} (Score: {match['score']:.2f})")

def test_vton():
    print("\n" + "=" * 60)
    print("2. TESTING VIRTUAL TRY-ON (IDM-VTON GRADIO API)")
    print("=" * 60)
    from gradio_client import Client, handle_file

    if not os.path.exists("test_person.jpg") or not os.path.exists("test_cloth.jpg"):
        print("[!] Sample images not found. Skipping VTON test.")
        return

    print("[*] Connecting to Hugging Face space 'Nymbo/Virtual-Try-On'...")
    client = Client("Nymbo/Virtual-Try-On")
    print("[*] Submitting test try-on job (person + garment)...")
    start = time.time()
    result = client.predict(
        dict={"background": handle_file("test_person.jpg"), "layers": [], "composite": None},
        garm_img=handle_file("test_cloth.jpg"),
        garment_des="A stylish white garment",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=20,
        seed=42,
        api_name="/tryon"
    )
    elapsed = time.time() - start
    print(f"[+] Success! Generated in {elapsed:.1f}s.")
    print(f"    Output image saved at: {result[0]}")

if __name__ == "__main__":
    test_search()
    test_vton()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

