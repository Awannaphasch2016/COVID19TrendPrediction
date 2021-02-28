import boto3
client = boto3.client('s3')

# add an object to s3 bucket
response = client.put_object()

