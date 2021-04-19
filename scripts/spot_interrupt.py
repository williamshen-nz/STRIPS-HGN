import argparse
import os
import socket
import tarfile
import time
from datetime import datetime

import boto3
import requests
from botocore.exceptions import ClientError

POLL_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spot interrupt script")
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    # Experiments directory
    parser.add_argument(
        "-e",
        "--experiments-dir",
        type=str,
        required=True,
        help="Path to the directory which contains the experiment results. "
        "The contents of this directory will be compressed and uploaded to S3",
    )

    # Args with defaults
    parser.add_argument(
        "--poll-delay",
        type=int,
        default=30,
        help="How often to poll for spot interruption",
    )

    # S3 Bucket
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="stripshgn-backup",
        help="Name of the S3 bucket to upload experiment results to.",
    )

    # SNS Topic
    parser.add_argument(
        "--sns-topic-arn",
        type=str,
        default="arn:aws:sns:us-east-2:650600419044:STRIPSHGN-Spot-Interrupt",
        help="SNS Topic to publish to when spot instance interrupt detected",
    )

    parser.add_argument("--region-name", type=str, default="us-east-2")

    return parser


class SpotInterruptHelper(object):
    def __init__(
        self,
        region_name: str,
        bucket: str,
        sns_topic_arn: str,
        poll_delay: int,
        experiments_dir: str,
    ):
        self.bucket = bucket
        self.sns_topic_arn = sns_topic_arn
        self.poll_delay = poll_delay
        self.experiments_dir = experiments_dir

        # Setup boto3
        self._session = boto3.Session(region_name=region_name)
        self._s3_client = self._session.client("s3")
        self._s3_resource = self._session.resource("s3")
        self._sns = self._session.client("sns")

    def _check_aws_connectivity(self):
        """ Check we can connect to S3 and have access to the SNS topic """
        try:
            self._s3_resource.meta.client.head_bucket(Bucket=self.bucket)
            print(f"Verified connectivity to S3 bucket s3://{self.bucket}")
        except ClientError as e:
            print(e)
            raise RuntimeError(
                f"s3://{self.bucket} does not exist or we don't have access"
            )

        try:
            self._sns.get_topic_attributes(TopicArn=self.sns_topic_arn)
            print(f"Verified connectivity to SNS topic {self.sns_topic_arn}")
        except ClientError as e:
            print(e)
            raise RuntimeError(
                f"Could not get topic attributes for {self.sns_topic_arn}"
            )

    def _publish_sns_message(self, message: str):
        """ Publish message to SNS topic """
        response = self._sns.publish(
            TopicArn=self.sns_topic_arn, Message=message
        )
        print(
            f"Published message to SNS topic {self.sns_topic_arn}. "
            f"MessageId = {response['MessageId']}"
        )

    def _upload_to_s3(self, body: bytes, key: str):
        """ Upload object to S3 """
        self._s3_client.put_object(Body=body, Bucket=self.bucket, Key=key)

    @staticmethod
    def _spot_interrupt_detected() -> bool:
        """ Whether a spot instance interrupt has been detected """
        response = requests.get(POLL_URL)
        return response.status_code != 404

    def _handle_spot_interrupt_detected(self):
        """
        Handle case where spot interrupt detected

        1. Compress all the results in the experiments directory
        2. Upload them to S3
        3. Publish a SNS message
        """
        if not os.path.exists(self.experiments_dir):
            raise ValueError(f"{self.experiments_dir} does not exist!")

        now_str = datetime.now().isoformat()

        # GZip the experiments result directory
        tar_name = f"backup_{now_str}.tar.gz"
        tar_fname_on_disk = os.path.join(f"/tmp/{tar_name}")
        tar = tarfile.open(tar_fname_on_disk, "w:gz")
        tar.add(self.experiments_dir)
        tar.close()
        print(f"GZipped {self.experiments_dir}")

        # Read tar from disk as bytes and delete the file
        tar_bytes = open(tar_fname_on_disk, "rb").read()

        # Upload to S3
        self._upload_to_s3(tar_bytes, tar_name)
        s3_path = f"s3://{self.bucket}/{tar_name}"
        print(f"Uploaded GZip to {s3_path}")

        # Publish SNS message
        message = (
            "WARNING! STRIPS-HGN Spot Instance Interrupt Detected.\n"
            f"Uploading contents of '{self.experiments_dir}' to {s3_path}\n\n"
            f"Hostname: {socket.gethostname()}\nTimestamp: {now_str}"
        )
        self._publish_sns_message(message)

    def _start_polling(self):
        """ Keep polling the instance-action page until the host shuts down """
        while True:
            # Poll the instance-action page
            now = datetime.now().isoformat()

            if self._spot_interrupt_detected():
                print(f"POLLED AT {now}, SPOT INTERRUPT DETECTED!!!")
                self._handle_spot_interrupt_detected()
            else:
                print(f"Polled at {now} - no interrupt detected")

            # Sleep
            print(f"Sleeping {self.poll_delay}s...")
            time.sleep(self.poll_delay)

    def start(self):
        """ Check AWS connectivity and start the polling """
        self._check_aws_connectivity()
        self._start_polling()


if __name__ == "__main__":
    # Parse CLI args and get helper
    args = get_parser().parse_args()
    helper = SpotInterruptHelper(**vars(args))
    helper.start()
