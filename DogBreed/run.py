from flask import Flask, render_template, url_for, request
from helper import *

app = Flask(__name__)

@app.route('/')
def home():

	return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		img_path = request.form['message']
		img_path1 = request.form['message']

	with urlopen(img_path) as f:
		img_path = io.BytesIO(f.read())



	VGG16 = models.vgg16(pretrained=True)

	n_inputs = 4096
	n_outputs = 133

	out_layer = nn.Linear(n_inputs, n_outputs)
	VGG16.classifier[6] = out_layer
	model_transfer = VGG16

	model_inference = load_checkpoint('checkpoint.pth', model_transfer)


	message = run_app(img_path, img_path1)

	ps, breeds = predict1(img_path, model_inference)

	message1 = breeds[0]
	message3 = breeds[1]
	message4 = ps[0]

	return render_template('result.html', message=img_path1, message1=message1, message2=message, message3=message3, message4=message4)



if __name__ == '__main__':
	app.run(debug=True)