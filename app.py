from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import os
from PIL import Image
import numpy as np
from pymongo import MongoClient
from werkzeug.utils import secure_filename

# Load face detection and recognition classifiers
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

app = Flask(__name__)
app.secret_key = "secret_key"

# Connect to MongoDB
client = MongoClient('mongodb+srv://admin:admin123@cluster0.ggnhlvo.mongodb.net/?retryWrites=true&w=majority')
db = client['Authorized_user']
collection = db['my_table']

label_to_name = {}
for user_data in collection.find({}, {"Name": 1}):
    if "Name" in user_data:
        label_to_name[user_data["_id"]] = user_data["Name"]


UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/generate_dataset', methods=['GET', 'POST'])
def generate_dataset():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        address = request.form['address']
        if name == "" or age == "" or address == "":
            flash('Please provide complete details of the user')
        else:
            # Insert user data into MongoDB
            id = collection.count_documents({}) + 1
            user_data = {"_id": id, "Name": name, "Age": age, "Address": address}
            collection.insert_one(user_data)

            # Your face detection and dataset generation logic here
            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            def face_cropped(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    return None
                for (x, y, w, h) in faces:
                    cropped_face = img[y:y + h, x:x + w]
                return cropped_face

            cap = cv2.VideoCapture(0)
            img_id = 0

            while True:
                ret, frame = cap.read()
                face = face_cropped(frame)
                if face is not None:
                    img_id += 1
                    face = cv2.resize(face, (200, 200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = f"data/user.{id}.{img_id}.jpg"
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Cropped face", face)
                    if cv2.waitKey(1) == 13 or int(img_id) == 200:
                        break
            cap.release()
            cv2.destroyAllWindows()
            flash('Generating dataset completed!!!')
        return redirect(url_for('home'))
    else:
        # Render the template containing the generate dataset form
        return render_template('generate_dataset.html')


# Define route for detecting faces
@app.route('/detect_faces')
def detect_faces():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            # Fetch user data from MongoDB
            user_data = db.my_table.find_one({"_id": id})
            print("ID:", id)
            print("User Data:", user_data)

            if user_data and "Name" in user_data:
                name = user_data["Name"]
            else:
                name = "UNKNOWN"

            if confidence > 50:
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Redirect to home page after face detection
    return redirect(url_for('home'))


# Define route for training classifier
@app.route('/train_classifier', methods=['POST'])
def train_classifier():
    data_dir = "data"
    # Error checking to ensure only image files are considered
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    # Show success message
    flash('Classifier training completed!')

    # Redirect to home page
    return redirect(url_for('home'))


@app.route('/users')
def users():
    # Pagination
    page = int(request.args.get('page', 1))
    per_page = 10  # Number of items per page
    total_users = collection.count_documents({})  # Total number of users
    total_pages = (total_users + per_page - 1) // per_page

    # Calculate offset and limit for database query
    offset = (page - 1) * per_page
    users = collection.find().skip(offset).limit(per_page)

    # Filtering
    filter_criteria = {}
    name = request.args.get('name')
    if name:
        filter_criteria['Name'] = {'$regex': f'.*{name}.*', '$options': 'i'}
    age = request.args.get('age')
    if age:
        filter_criteria['Age'] = int(age)

    if filter_criteria:
        users = collection.find(filter_criteria).skip(offset).limit(per_page)

    return render_template('users.html', users=users, page=page, per_page=per_page, total_pages=total_pages)


def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return cv2.putText(image, "No faces detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2, cv2.LINE_AA)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, pred = clf.predict(face)
        confidence = int(100 * (1 - pred / 300))

        # Get the name from the label_to_name dictionary
        name = label_to_name.get(label, "UNKNOWN")

        # Print the predicted label and name
        print(f"Predicted label: {label}, Name: {name}")

        # Display the name for each face individually
        cv2.putText(image, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  # Use secure_filename to ensure file name is safe
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            img = recognize_faces(img)
            cv2.imwrite("static/result.jpg", img)
            return render_template('result.html', result_image='static/result.jpg')
    return render_template('upload_image.html')


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
