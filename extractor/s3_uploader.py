"""
Upload local figure crops to S3 and return their public URLs.
"""

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from django.conf import settings


def upload_to_s3(local_url: str) -> str:
    """
    Upload a local /media/figures/ file to S3 and return the public URL.

    local_url: Django media URL, e.g. '/media/figures/abc123.png'
    Returns:   'https://<bucket>.s3.<region>.amazonaws.com/figures/abc123.png'
    """
    relative = local_url.removeprefix('/media/')
    local_path = settings.MEDIA_ROOT / relative

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    s3 = boto3.client(
        's3',
        region_name=settings.AWS_S3_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    try:
        s3.upload_file(
            str(local_path),
            settings.AWS_S3_BUCKET_NAME,
            relative,
            ExtraArgs={'ContentType': 'image/png'},
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"S3 upload failed: {e}") from e

    return (
        f"https://{settings.AWS_S3_BUCKET_NAME}"
        f".s3.{settings.AWS_S3_REGION}.amazonaws.com/{relative}"
    )
