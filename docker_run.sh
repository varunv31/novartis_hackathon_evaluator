IMAGE_NAME="aicrowd/novartis_hackathon_evaluator"
IMAGE_TAG="v1"
UPLOAD_FOLDER="/tmp"
UPLOAD_EXTRACT_FOLDER="/tmp"
UNIQUE_ACCESS_KEY="QUlDUk9XRF9HUkFERVIK"
GROUND_TRUTH_DATA_DIRECTORY="data/ground_truth"
SSL_ENABLE="True"
SSL_PRIVATEKEY_FILE="/path/to/private.key"
SSL_CERTIFICATE_FILE="/path/to/certificate.crt"

sudo docker run -p 5000:5000 \
-v $PWD/${GROUND_TRUTH_DATA_DIRECTORY}:/home/aicrowd/data/ground_truth \
-v $PWD/${SSL_PRIVATEKEY_FILE}:/home/aicrowd/certificate/key.pem \
-v $PWD/${SSL_CERTIFICATE_FILE}:/home/aicrowd/certificate/cert.pem \
-e UPLOAD_FOLDER=${UPLOAD_FOLDER} \
-e UPLOAD_EXTRACT_FOLDER=${UPLOAD_EXTRACT_FOLDER} \
-e UNIQUE_ACCESS_KEY=${UNIQUE_ACCESS_KEY} \
-e SSL_ENABLE=${SSL_ENABLE} \
-it ${IMAGE_NAME}:${IMAGE_TAG}
