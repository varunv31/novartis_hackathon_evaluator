IMAGE_NAME="aicrowd/novartis_hackathon_evaluator"
IMAGE_TAG="v1"
REPO2DOCKER="$(which aicrowd-repo2docker)"

sudo ${REPO2DOCKER} --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name ${IMAGE_NAME}:${IMAGE_TAG} \
  --debug .
