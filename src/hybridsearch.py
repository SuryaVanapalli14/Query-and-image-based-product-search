import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize connection to Pinecone
api_key = os.getenv("PINECONE_API_KEY", "")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set. Please add it to your .env file.")
pc = Pinecone(api_key=api_key)

# Configure serverless spec
cloud = os.environ.get('PINECONE_CLOUD', 'aws')
region = os.environ.get('PINECONE_REGION', 'us-east-1')
spec = ServerlessSpec(cloud=cloud, region=region)

# Index name
index_name = os.getenv("PINECONE_INDEX_NAME", "hybrid-image-search")

# Check and create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=512,
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to index
index = pc.Index(index_name)
print("Index Stats:", index.describe_index_stats())

# Load the dataset from Hugging Face datasets hub
fashion = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train"
)
images = fashion["image"]

# Load the CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)

# Perform a text-based query
def search_with_text_query(query):
    print(f"\nPerforming search for query: {query}")
    dense = model.encode(query).tolist()
    result = index.query(
        top_k=5,
        vector=dense,
        include_metadata=True
    )

    # Fetch images dynamically using the IDs from the query results
    result_images = [images[int(r["id"])] for r in result["matches"]]

    # Display results using matplotlib
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(result_images):
        plt.subplot(1, len(result_images), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Result {i + 1}")
    plt.tight_layout()
    plt.show()

# Perform an image-based query from file
def search_with_image_file(image_path):
    print("\nPerforming search with an image file...")
    # Load and process the image
    input_image = Image.open(image_path).convert("RGB").resize((224, 224))
    dense = model.encode(input_image, convert_to_tensor=True).cpu().tolist()

    result = index.query(
        top_k=5,
        vector=dense,
        include_metadata=True
    )

    # Fetch images dynamically using the IDs from the query results
    result_images = [images[int(r["id"])] for r in result["matches"]]

    # Display results using matplotlib
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(result_images):
        plt.subplot(1, len(result_images), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Result {i + 1}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Perform text-based search
    text_query = "dark blue french connection jeans for men"
    search_with_text_query(text_query)

    # Perform image-based search from file
    image_file_path = "test_cloth.jpg"
    if os.path.exists(image_file_path):
        search_with_image_file(image_file_path)
