import os
import requests
import base64
import cv2
import numpy as np
from flask import Flask, request, send_from_directory
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from gradio_client import Client as GradioClient, file
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
import matplotlib.pyplot as plt
import torch
import time
from dotenv import load_dotenv
import shutil

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN) else None

# Gradio Client for Virtual Try-On API
gradio_client = GradioClient("Nymbo/Virtual-Try-On")

# Ngrok URL
NGROK_URL = os.getenv("NGROK_URL", "").rstrip("/")

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY", "pcsk_2XvhS_F8DBNuXvffNAUVDRnYUccr21mwaPQ7Xwpa7wJLW8Ztgp8a2jpGeQKVMFJ4THENJ")
pc = Pinecone(api_key=api_key)
index_name = os.getenv("PINECONE_INDEX_NAME", "hybrid-image-search")
existing_indexes = pc.list_indexes().names()
if index_name not in existing_indexes:
    pc.create_index(index_name, dimension=512, metric='dotproduct', spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

# Load dataset and models
fashion = load_dataset("ashraq/fashion-product-images-small", split="train")
images = fashion["image"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32', device=device)

# In-memory session tracking
user_sessions = {}

@app.route("/", methods=["GET"])
def index_route():
    return "This is the virtual try-on chatbot API.", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    sender_number = request.form.get("From")
    user_message = request.form.get("Body").strip()
    media_url = request.form.get("MediaUrl0")
    resp = MessagingResponse()

    if sender_number not in user_sessions:
        user_sessions[sender_number] = {}
        resp.message("Welcome to AI Fashion Assistant!\nTo Get Started Choose From Below Options:\n0. Virtual Try-On\n1. Text Search\n2. Image Search\nTo return back to main menu after results send any meesage.")
        return str(resp)

    if "menu_option" not in user_sessions[sender_number]:
        if user_message == "0":
            user_sessions[sender_number]["menu_option"] = "try_on"
            resp.message("Send your image on which you want to try clothes for Virtual Try-On! ✨")
        elif user_message == "1":
            user_sessions[sender_number]["menu_option"] = "text_search"
            resp.message("Enter your search query to explore our product database! 🔍")
        elif user_message == "2":
            user_sessions[sender_number]["menu_option"] = "image_search"
            resp.message("Send an image to search our product database and find what you’re looking for! 🔎")
        else:
            resp.message("Invalid choice. Please choose an option from below: \n\n0. Virtual Try-On\n1. Text Search\n2. Image Search")
        return str(resp)

    if user_sessions[sender_number]["menu_option"] == "try_on":
        handle_try_on(sender_number, media_url, resp)
    elif user_sessions[sender_number]["menu_option"] == "text_search":
        handle_text_search(sender_number, user_message, resp)
    elif user_sessions[sender_number]["menu_option"] == "image_search":
        handle_image_search(sender_number, media_url, resp)

    return str(resp)


def handle_try_on(sender, media_url, resp):
    session = user_sessions[sender]
    if "person_image" not in session:
        if media_url:
            session["person_image"] = media_url
            resp.message("Great! Now Send the garment image which you want to try.")
        else:
            resp.message("Send an image to start.")
    elif "garment_image" not in session:
        if media_url:
            session["garment_image"] = media_url
            person_image_url = session["person_image"]
            garment_image_url = media_url

            try_on_image_url = send_to_gradio(person_image_url, garment_image_url)
            if try_on_image_url:
                send_media_message(sender, try_on_image_url)
                resp.message("Here is your virtual try-on result! Returning to menu.")
            else:
                resp.message("Sorry, something went wrong with the try-on process. Returning to menu.")

            del user_sessions[sender]
        else:
            resp.message("Send the garment image.")


def send_to_gradio(person_image_url, garment_image_url):
    person_image_path = download_image(person_image_url, 'person_image.jpg')
    garment_image_path = download_image(garment_image_url, 'garment_image.jpg')

    if person_image_path is None or garment_image_path is None:
        print("Error: One of the images could not be downloaded.")
        return None
    try:
        # Interact with the Gradio API using the client
        result = gradio_client.predict(
            dict={"background": file(person_image_path), "layers": [], "composite": None},
            garm_img=file(garment_image_path),
            garment_des="A cool description of the garment",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )
        if result and len(result) > 0:
            try_on_image_path = result[0]
            print(f"Generated try-on image path: {try_on_image_path}")
            # Ensure the static directory exists
            static_dir = 'static'
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                print(f"Created directory: {static_dir}")

            # Make sure the path exists
            if os.path.exists(try_on_image_path):
                # Convert the image to PNG format and save it
                img = cv2.imread(try_on_image_path)
                target_path_png = os.path.join(static_dir, 'result.png')
                cv2.imwrite(target_path_png, img)
                print(f"Image saved to: {target_path_png}")
                # Return the public URL for the image as PNG
                return f"{NGROK_URL}/static/result.png"
            else:
                print(f"Image not found at: {try_on_image_path}")
                return None
        print("No image returned from the API.")
        return None
    except Exception as e:
        print(f"Error interacting with Gradio API: {e}")
        return None


def copy_to_static(source_path_or_url, filename):
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    destination = os.path.join(static_dir, filename)

    try:
        # If source is a file path
        if os.path.exists(source_path_or_url):
            shutil.copy(source_path_or_url, destination)
        else:  # If source is a URL
            response = requests.get(source_path_or_url)
            with open(destination, "wb") as f:
                f.write(response.content)
        print(f"File copied to static directory as {filename}")
        return filename
    except Exception as e:
        print(f"Error copying file to static: {e}")
        return None


def handle_text_search(sender, query, resp):
    results = search_with_text_query(query)
    if results:
        static_result = save_static_image(results[0], "result_text_search.png")
        send_media_message(sender, f"{NGROK_URL}/static/{static_result}")
        resp.message("*Your result is ready!* 🎉\nCheck it out below: 👇")
    else:
        resp.message("No results found. Returning to menu.")
    del user_sessions[sender]

def handle_image_search(sender, media_url, resp):
    if media_url:
        # Download the image using the same logic as in the try-on option
        image_path = download_image(media_url, "search_image.jpg")
        if image_path:  # Ensure the image was successfully downloaded
            # Perform search with the downloaded image
            results = search_with_image_file(image_path)
            if results:
                # Save the first result to the static directory
                static_result = save_static_image(results[0], "result_image_search.png")
                # Send the result back via WhatsApp
                send_media_message(sender, f"{NGROK_URL}/static/{static_result}")
                resp.message("*Your result is ready!* 🎉\nCheck it out below: 👇")
            else:
                resp.message("No results found. Returning to menu.")
        else:
            # If image download failed, log the issue and inform the user
            print(f"Failed to download image from URL: {media_url}")
            resp.message("We couldn't process your image. Please try again.")
        # Clear session after completing the request
        del user_sessions[sender]
    else:
        # If no media_url is provided, inform the user
        resp.message("We didn't receive an image. Please try sending your image again.")




def search_with_text_query(query):
    vector = clip_model.encode(query).tolist()
    result = index.query(top_k=5, vector=vector, include_metadata=True)
    return [images[int(r["id"])] for r in result["matches"]]

def search_with_image_file(image_path):
    print(f"Performing search with image file: {image_path}")
    try:
        input_image = Image.open(image_path).convert("RGB").resize((224, 224))
        dense = clip_model.encode(input_image, convert_to_tensor=True).cpu().tolist()
        result = index.query(top_k=5, vector=dense, include_metadata=True)
        return [images[int(r["id"])] for r in result["matches"]]
    except Exception as e:
        print(f"Error during image search: {e}")
        return None

def save_static_image(image, filename):
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    path = os.path.join(static_dir, filename)
    cv2.imwrite(path, np.array(image))
    print(f"Image saved to static directory as {filename}")
    return filename

def download_image(media_url, filename):
    try:
        # Extract Message SID and Media SID from the URL
        message_sid = media_url.split('/')[-3]
        media_sid = media_url.split('/')[-1]

        # Log the message and media SIDs
        print(f"Message SID: {message_sid}, Media SID: {media_sid}")

        # Use Twilio client to fetch the media resource
        media = client.api.accounts(TWILIO_ACCOUNT_SID).messages(message_sid).media(media_sid).fetch()

        # Construct the actual media URL
        media_uri = media.uri.replace('.json', '')
        image_url = f"https://api.twilio.com{media_uri}"

        # Log the full URL being used for the image download
        print(f"Downloading image from: {image_url}")

        # Download the image with proper authorization (using Basic Auth)
        response = requests.get(image_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

        if response.status_code == 200:
            # Save the image locally
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded successfully as {filename}.")
            return filename
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
            return None
    except Exception as err:
        print(f"Error downloading image from Twilio: {err}")
        return None

def send_media_message(to, media_url):
    if client:
        client.messages.create(from_=os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886"), body="Your result:", media_url=[media_url], to=to)
    else:
        print(f"Twilio client not initialized. Cannot send message to {to}")


@app.route('/static/<path:filename>')
def serve_static_file(filename):
    file_path = os.path.join('static', filename)
    # Check if the file exists and serve with the correct Content-Type
    if os.path.exists(file_path):
        return send_from_directory('static', filename, mimetype='image/png')
    else:
        print(f"File not found: {filename}")
        return "File not found", 404


if __name__ == "__main__":
    app.run(port=8080)