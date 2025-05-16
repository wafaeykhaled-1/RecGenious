# 🧥 Fashion Image Recommendation System

A content-based fashion recommendation system built with Flask and EfficientNet. Users can browse products by category and upload an image to receive visually similar items.

---

## 📌 Project Overview

This system provides:

- 👗 Category-wise product browsing
- 🧠 Deep learning-based image similarity (EfficientNetB0)
- 🧾 Product metadata from CSV files
- 👤 Optional filtering by gender (only if >10 items exist per gender)

---

## 🧠 How It Works

- Product metadata is loaded from `styles.csv`
- Images are matched using precomputed feature vectors in `image_features66.npz`
- Uploaded image is processed using **EfficientNetB0**, and the most visually similar products are returned
- On category pages, a gender filter appears only if there are enough data points (more than 10 items per gender)

---

## 📁 File Structure

Since this repo contains only code and CSV files, here’s what each file is for:

├── app.py # Main Flask app
├── styles.csv # Metadata of fashion products
├── images.csv # Filename-to-link mapping (optional)
├── templates/ # HTML templates (Jinja2)
├── static/ # Static files (images, CSS, etc.)
│ ├── uploads/ # Where user-uploaded images are stored
│ └── images/ # Product images (from Kaggle)

## 📥 Required Downloads

### 🔹 1. Download Precomputed Features

Download the image features file (`image_features66.npz`) from OneDrive:

📦 [Download image_features66.npz](https://1drv.ms/u/c/40d0a0f9d35b2ee8/EVuv3MBcnGVClGId7VudkGUB4FaAXTaB80x2Ar0owRfhGQ?e=rWQbb9)

Place the file in the **root directory** of the project.

---

### 🔹 2. Download Images from Kaggle

You must download the original product images from Kaggle to run the project correctly:

📸 Dataset: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

**Steps:**
1. Download and unzip the dataset.
2. Move all image files (`.jpg`) to:  
   `static/images/`

Your structure should look like:

static/images/10000.jpg
static/images/10001.jpg

---

## 🚀 Run the App

Install the requirements and start the server:

```bash
pip install -r requirements.txt
python app.py
