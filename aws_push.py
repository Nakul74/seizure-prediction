import boto3
import os

service = 's3'
access_key = 'AKIAYELTLM3DJ6YDPWS3'
secret_key = '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'
region = 'us-east-1'
bucket_name = 'epilepsypatient'

def upload_file_to_s3(s3_client,file_name,s3_file_name):
    s3_client.upload_file(file_name, bucket_name, s3_file_name)
    
    
def get_aws_client():
    s3_client = boto3.client(service_name = service, 
                            aws_access_key_id = access_key,
                            aws_secret_access_key = secret_key,
                            region_name = region)
    return s3_client

def get_public_links(s3_client):
    files_dict = {}

    try:
        # List objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)

        # Iterate through the objects
        for obj in response.get('Contents', []):
            # Get the object key (file name)
            file_name = obj['Key']

            # Generate a public link to download the file
            link = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': bucket_name, 'Key': file_name},
                ExpiresIn=604800  # Link expiration time (in seconds)
            )

            # Add the file name and link to the dictionary
            files_dict[file_name] = link

    except Exception as e:
        print(f"An error occurred: {e}")

    return files_dict

if __name__ == '__main__':
    s3_client = get_aws_client()
    files_dict = get_public_links(s3_client)
    print(files_dict)

