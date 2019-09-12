# Novartis Hackathon Evaluator

The evaluator runs as a flask application which expose single endpoint `/evaluate` (POST).

The expected parameters are:

1. `submission`: Zip file of the submission submitted by participant
2. `key`: Authentication key used by AIcrowd grader (basic protection)

## cURL Example
```
curl -XPOST http://localhost:5000/evaluate -F "submission=@/tmp/submission.zip" -F "key=QUlDUk9XRF9HUkFERVIK"
```

## Python Example
```
import json
import requests
submission = open('/tmp/submission.zip', 'rb')
r = requests.post('http://127.0.0.1:5000/evaluate', data = {"key":"QUlDUk9XRF9HUkFERVIK"}, files={"submission": ("test.zip", submission)})
json.loads(r.text)
#{'message': 'Submission Evaluated', 'meta': {'Task-1-Score': 1, 'Task-2-Score': 1.0, 'Task-3-Score': 0.5193820017607189, 'Task-4-Score': 0.7303686378204532, 'Task-5-Score': 0.26963136217954675}, 'score': 0.6518883280923063, 'score_secondary': 1, 'success': True}
```

## Run evaluator on machine
```
>> ./docker_build.sh
>> ./docker_run.sh
```
