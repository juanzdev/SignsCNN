import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append("src")
import model
import cv2
import numpy as np
#!flask/bin/python
from flask import Flask, render_template, jsonify, abort, make_response, request, url_for, redirect, json, send_from_directory
from werkzeug.utils import secure_filename
import random, string

app = Flask(__name__)

num_channels = 1
img_size = 28
img_size_flat = img_size * img_size * num_channels

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
sess = tf.Session()
# restore trained data
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    print(x)
    y,y_conv,y_conv_cls,variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "snapshots/snp_8624")

def convolutional(input):
    return sess.run(y, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'ws/uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))



tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/api/sign/upload', methods=['POST'])
def upload():
    print("HOLA")
    # Get the name of the uploaded file
    files = request.files.getlist("file[]")
    signs = {}
    #x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    print(files)
    #input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(None, img_size_flat)
    #output = convolutional(input)
    #return jsonify(results=[output])
    for file in files:
        print("loop")
        print(file)
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            print(filename)
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
            #image = cv2.imread(filename)
            print("IMAGE")
            print(image)

            imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #sub_image = imgray[20:350, 25:195]
            equ = cv2.equalizeHist(imgray)
            equ_resize = cv2.resize(equ,(img_size,img_size))

            images = []
            print(equ_resize.shape)
            images.append(equ_resize)
            images = np.array(images)
            train_batch_size = 1
            img_size_flat = img_size * img_size * num_channels
            print(img_size_flat)
            x_batch = images;
            x_batch = x_batch.reshape(train_batch_size, img_size_flat)


            #x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
            # Move the file form the temporal folder to
            # the upload folder we setup

            # dummy value for sign image
            #sings[filename] = random_generator()

            # Here comes the logic neural network
            #input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
            output2 = convolutional(x_batch)
            substract_output = np.zeros(2)
            substract_output[0] = output2[1]
            substract_output[1] = output2[2]
            print(substract_output)
            signs[filename] = output2
            #return jsonify(results=[output1, output2])

            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Redirect the user to the uploaded_file route, which
            # will basically show on the browser the uploaded file
            print ('File Name: ' + filename)
    
    return jsonify(signs)

@app.route('/api/messages', methods = ['POST'])
def api_message():
    if request.headers['Content-Type'] == 'text/plain':
        return "Text Message: " + request.data

    elif request.headers['Content-Type'] == 'application/json':
        return "JSON Message: " + json.dumps(request.json)

    elif request.headers['Content-Type'] == 'application/octet-stream':
        f = open('./binary', 'wb')
        f.write(request.data)
        f.close()
        return "Binary message written!"

    else:
        return "415 Unsupported Media Type ;)"

def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id=task['id'], _external=True)
        else:
            new_task[field] = task[field]
    return new_task

@app.route('/api/todo/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': [make_public_task(task) for task in tasks]})

@app.route('/api/todo/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/api/todo/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201

if __name__ == '__main__':
    app.run(
        host="127.0.0.1",
        port=int("8080"),
        debug=True
    )