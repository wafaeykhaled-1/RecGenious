from flask import Flask, render_template, request, redirect, url_for, flash,session
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input, EfficientNetB0
from keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --------------------- Load and prepare data ---------------------

# Read CSV files
styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')
images_df = pd.read_csv('images.csv', on_bad_lines='skip')

# Load extracted image features
image_features_data = np.load('image_features66.npz')
features = image_features_data['features']
image_ids = image_features_data['ids']

# Create DataFrame for image features
features_df = pd.DataFrame(features, index=image_ids)
features_df.index = features_df.index.astype(str)  # Ensure IDs are strings

# Create lookup for product information
styles_df['id'] = styles_df['id'].astype(str)
info_lookup = styles_df.set_index('id').to_dict(orient='index')

# Map filenames to links
images_df['filename'] = images_df['filename'].astype(str)
images_df['link'] = images_df['link'].astype(str)
id_to_link = dict(zip(images_df['filename'], images_df['link']))

# --------------------- Load EfficientNet model ---------------------

# Load EfficientNetB0 model without top layer (for feature extraction)
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# --------------------- Feature extraction function ---------------------

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array)[0]

# --------------------- Find similar images ---------------------

def get_similar_images(input_path, features_df, top_k=5):
    input_features = extract_features(input_path).reshape(1, -1)
    distances = euclidean_distances(input_features, features_df.values)[0]
    top_indices = np.argsort(distances)[:top_k]
    return [features_df.index[i] for i in top_indices]

# --------------------- Home Route ---------------------

@app.route('/')
def index():
    # Get top 10 most frequent categories
    categories_data = styles_df['articleType'].value_counts().head(10)

    categories = []
    for category_name, _ in categories_data.items():
        # Get representative image for each category
        category_info = styles_df[styles_df['articleType'] == category_name].iloc[0]
        image_filename = f"{category_info['id']}.jpg"

        category_info_dict = {
            'name': category_name,
            'image_filename': image_filename,
        }
        categories.append(category_info_dict)

    return render_template('index.html', categories=categories)


# --------------------- Category Route ---------------------

@app.route('/category/<category_name>')
def category(category_name):
    category_data = styles_df[styles_df['articleType'] == category_name]

    gender_counts = category_data['gender'].value_counts()
    valid_genders = gender_counts[gender_counts > 10].index.tolist()
    genders = valid_genders if len(valid_genders) > 1 else []

    selected_gender = request.args.get('gender')

    if selected_gender:
        category_data = category_data[category_data['gender'] == selected_gender]

    category_data = category_data.head(50)

    products = []
    for _, row in category_data.iterrows():
        products.append({
            'image': f"/static/images/{row['id']}.jpg",
            'name': row['productDisplayName'],
            'price': round(np.random.uniform(100, 500), 2),
            'type': row['articleType'],
            'color': row['baseColour']
        })

    return render_template(
        'category.html',
        category=category_name,
        products=products,
        genders=genders,
        selected_gender=selected_gender
    )



# --------------------- Upload and Recommendation Route ---------------------

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded image
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Find similar images
            similar_ids = get_similar_images(filepath, features_df)

            results = []
            for img_id in similar_ids:
                clean_id = img_id.replace('.jpg', '')
                style_info = info_lookup.get(clean_id)

                if style_info:
                    results.append({
                        'path': f"/static/images/{img_id}",
                        'name': style_info['productDisplayName'],
                        'type': style_info['articleType'],
                        'color': style_info['baseColour'],
                        'price': f"{np.random.randint(200, 1000)}"
                    })

            # Save results to session
            session['results'] = results
            session['original'] = filename

            # Show success message only when image is uploaded
            flash('Image uploaded successfully!', 'success')
            return redirect(url_for('results'))

    return render_template('index.html')


@app.route('/results')
def results():
    # Retrieve results and original image name from session
    original = session.get('original')
    results = session.get('results')

    # If no results in session, return error or redirect back
    if not results or not original:
        flash("No results found.", "danger")
        return redirect(url_for('upload'))

    return render_template('results.html', original=original, results=results)

# --------------------- Run the App ---------------------

if __name__ == '__main__':
    app.run(debug=True)
