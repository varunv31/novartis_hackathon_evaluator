# coding: utf-8
import os
import pandas as pd
import numpy as np
import tempfile
import warnings
import zipfile
from flask import Flask
from flask import request, jsonify
from shutil import copyfile
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from werkzeug.utils import secure_filename
from evaluator import NovartisHackathonEvaluator

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp')
UPLOAD_EXTRACT_FOLDER = os.getenv('UPLOAD_EXTRACT_FOLDER', '/tmp')
UNIQUE_ACCESS_KEY = os.getenv('UNIQUE_ACCESS_KEY', 'QUlDUk9XRF9HUkFERVIK')

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

        evaluator = NovartisHackathonEvaluator("data/ground_truth", debug=False)
        score_object = evaluator.evaluate(tmpdirname)
        return jsonify({**score_object, **{"success": True, "message": "Submission Evaluated"}}), 200

    """
    # Benchmark
    import tqdm
    for k in tqdm.tqdm(range(1000)):
        score_object = evaluator.evaluate("data/submission")
    """

if __name__ == "__main__":
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn
    app.run(host='0.0.0.0')
