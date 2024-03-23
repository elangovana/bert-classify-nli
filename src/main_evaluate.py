import argparse
import json
import logging
import os
import sys

import boto3


class Evaluate:

    def __init__(self):
        self.client = boto3.client('sagemaker-runtime')

    def run(self, json_file, model_endpoint, output_dir):
        items = self._parse_json_file(json_file)

        batch_size = 5
        result = []

        correct_predictions = 0
        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]

            response = self.client.invoke_endpoint(
                EndpointName=model_endpoint,
                Body=json.dumps(batch).encode("utf-8"),
                ContentType='text/json',
                Accept='text/json'
            )

            predictions = json.loads(response["Body"].read().decode("utf-8"))

            for bi, p in zip(batch, predictions):
                bi["predicted_label"] = p["label"]
                bi["predicted_confidence"] = p["confidence"]
                if bi["predicted_label"] == bi["label"]: correct_predictions += 1

            result.extend(batch)

        accuracy = 100 * correct_predictions / len(result)
        self._logger.info(f"% Accuracy == {accuracy}")

        # Dump predictions
        os.makedirs(output_dir, exist_ok=True)
        output_file_prefix = os.path.splitext(os.path.basename(json_file))[0]
        output_file = os.path.join(output_dir, f"{output_file_prefix}_predictions.json")
        self._logger.info(f"Writing output to {output_file}")
        with open(output_file, "w") as f:
            json.dump(result, f)

    def _parse_json_file(self, json_file):
        items = []
        with open(json_file) as f:
            for i, l in enumerate(f):
                data = json.loads(l)
                item = {"premise": data["sentence1"],
                        "hypothesis": data["sentence2"],
                        "label": data["gold_label"]
                        }

                if item["label"] == "-":
                    self._logger.info(f"Missing label in idx {i}.. hence skipping")
                    continue

                items.append(item)
        return items

    @property
    def _logger(self):
        return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testjson",
                        help="The input file to evaluate")

    parser.add_argument("outputdir",
                        help="The output directory to write to")

    parser.add_argument("modelendpoint",
                        help="The name of the endpoint")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    Evaluate().run(args.testjson,  args.modelendpoint,args.outputdir)


if "__main__" == __name__:
    main()
