#--------------------------------------------------
# Import Requirements
#--------------------------------------------------
import os
from flask import Flask
from flask_socketio import SocketIO
from flask_failsafe import failsafe

socketio = SocketIO()

#--------------------------------------------------
# Create a Failsafe Web Application
#--------------------------------------------------
@failsafe
def create_app(debug=False):
	app = Flask(__name__)

	# This will prevent issues with cached static files
	app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
	app.config['UPLOAD_FOLDER'] = 'flask_app/static/main/input_images/container'
	app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
	app.debug = debug
	# The secret key is used to cryptographically-sign the cookies used for storing the session data.
	app.secret_key = 'AKWNF1231082fksejfOSEHFOISEHF24142124124124124iesfhsoijsopdjf'
	# ----------------------------------------------

	#from .utils.database.database import database
	#db = database()

	socketio.init_app(app)

	with app.app_context():
		from . import routes
		return app
