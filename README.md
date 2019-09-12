# Novartis Hackathon Evaluator

The evaluator runs as a flask application which expose single endpoint `/evaluate` (POST).

The expected parameters are:

1. `submission`: Zip file of the submission submitted by participant
2. `key`: Authentication key used by AIcrowd grader (basic protection)

## cURL Example
```
curl -XPOST http://localhost:5000/evaluate -F "submission=@/tmp/submission.zip" -F "key=QUlDUk9XRF9HUkFERVIK"
```


