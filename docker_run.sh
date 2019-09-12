IMAGE_NAME="aicrowd/novartis_hackathon_evaluator"
IMAGE_TAG="v1"

sudo docker run -p 5000:5000 -it ${IMAGE_NAME}:${IMAGE_TAG}
