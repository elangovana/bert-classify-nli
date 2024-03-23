import argparse
import datetime
import logging
import os.path
import sys

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor


def run_processing_job(role, s3_data_file, s3_output_base, endpoint_name):
    pytorch_processor = PyTorchProcessor(
        py_version="py38",
        framework_version='1.12.0',
        role=role,
        instance_type='ml.m5.xlarge',
        instance_count=1,
        base_job_name='frameworkprocessor-PT'
    )

    job_name = "snli-evaluate-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    s3_output_uri = "{}/{}".format(s3_output_base.rstrip("/"), job_name)

    entry_point_file = "main_evaluate.py"
    sm_input_data_dir = '/opt/ml/processing/input'
    sm_output_dir = '/opt/ml/processing/output'
    pytorch_processor.run(
        code=entry_point_file,
        source_dir=os.path.dirname(__file__),
        arguments=[
            os.path.join(sm_input_data_dir, s3_data_file.split("/")[-1]),
            sm_output_dir,
            endpoint_name
        ],
        inputs=[
            ProcessingInput(
                input_name='data',
                source=s3_data_file,
                destination=sm_input_data_dir
            )
        ],

        outputs=[
            ProcessingOutput(output_name='data_structured',
                             source=sm_output_dir,
                             destination=s3_output_uri)
        ],

        job_name=job_name,
        wait=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role",
                        help="The sagamaker role to use", required=True)

    parser.add_argument("--s3input",
                        help="The  s3 input file", required=True)

    parser.add_argument("--s3output",
                        help="The output s3 path to write to", required=True)

    parser.add_argument("--modelendpoint",
                        help="The name of the endpoint")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    run_processing_job(args.role, args.s3input, args.s3output, args.modelendpoint)


if "__main__" == __name__:
    main()
