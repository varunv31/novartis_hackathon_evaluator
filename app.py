# coding: utf-8
import os
import pandas as pd
import numpy as np
import tempfile
import warnings
import zipfile
from distutils.util import strtobool
from flask import Flask
from flask import request, jsonify
from OpenSSL import SSL
from shutil import copyfile
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from werkzeug.utils import secure_filename
from evaluator import NovartisHackathonEvaluator

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp')
UPLOAD_EXTRACT_FOLDER = os.getenv('UPLOAD_EXTRACT_FOLDER', '/tmp')
UNIQUE_ACCESS_KEY = os.getenv('UNIQUE_ACCESS_KEY', 'QUlDUk9XRF9HUkFERVIK')
GROUND_TRUTH_DATA_FOLDER = os.getenv("GROUND_TRUTH_DATA_FOLDER", "data/ground_truth")
SSL_ENABLE = strtobool(os.getenv("SSL_ENABLE", "False"))
SSL_PRIVATEKEY_FILE = os.getenv("SSL_PRIVATEKEY_FILE", "certificate/key.pem")
SSL_CERTIFICATE_FILE = os.getenv("SSL_CERTIFICATE_FILE", "certificate/cert.pem")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/evaluate', methods = ['POST'])
def evaluate():
    if 'key' not in request.form or request.form['key'] != UNIQUE_ACCESS_KEY:
        return jsonify({"success": False, "message": "Authentication Failed"}), 403

    if 'submission' not in request.files:
        return jsonify({"success": False, "message": "Submission file not present in post request"}), 500

    submission_zip = request.files['submission']
    with tempfile.TemporaryDirectory(dir=UPLOAD_EXTRACT_FOLDER) as tmpdirname:
        filename = secure_filename(submission_zip.filename)
        filepath = tmpdirname + '/' + filename

        submission_zip.save(filepath)
        if not zipfile.is_zipfile(filepath):
            return jsonify({"success": False, "message": "Not a zip file"}), 500

        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
                os.unlink(filepath)
        except Exception as e:
            return jsonify({"success": False, "message": "Extraction of zip file failed with error: %s" % e}), 500

        evaluator = NovartisHackathonEvaluator(GROUND_TRUTH_DATA_FOLDER, debug=False)
        try:
            score_object = evaluator.evaluate(tmpdirname)
        except Exception as e:
            error_message = str(e)
            return jsonify({"success": False, "message": error_message}), 500
        return jsonify({**score_object, **{"success": True, "message": "Submission Evaluated"}}), 200

if __name__ == "__main__":
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    if not SSL_ENABLE:
        app.run(host='0.0.0.0')
    else:
        context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
        context.use_privatekey_file(SSL_PRIVATEKEY_FILE)
        context.use_certificate_file(SSL_CERTIFICATE_FILE)
        app.run(host='0.0.0.0', ssl_context=context)
