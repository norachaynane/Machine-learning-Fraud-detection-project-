# Fraud Detection using Machine Learning


## Project Description
The goal of this project is to use Amazon SageMaker to train both a supervised and an unsupervised machine learning model, which can then be deployed through SageMaker-managed endpoints. If you have labeled data, such as transactions that have been annotated as fraudulent or legitimate, the project includes instructions on how to train a gradient boosted decision tree model using XGBoost. The process also addresses the common challenge of imbalanced data by utilizing data upsampling and adjusting the positive and negative weight balance.



If you have labels for your data, for example if some of the transactions have been annotated as fraudulent and some as legitimate, then you can train a supervised learning model to learn to discern the two classes. In this project, we provide a recipe to train a gradient boosted decision tree model using [XGBoost on Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html). The supervised model training process also handles the common issue of working with highly imbalanced data in fraud detection problems. The project addresses this issue into two ways by 1) implementing data upsampling using the "imbalanced-learn" package, and 2) using scale position weight to control the balance of positive and negative weights.

If you don't have labelled data or if you want to augment your supervised model predictions with an anomaly score from an unsupervised model, then the project also trains a [RandomCutForest](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html) model using Amazon SageMaker. The RandomCutForest algorithm is trained on the entire dataset, without labels, and takes advantage of the highly imbalanced nature of fraud datasets, to predict higher anomaly scores for the fraudulent transactions in the dataset.

Both of the trained models are deployed to Amazon SageMaker managed real-time endpoints that host the models and can be invoked to provide model predictions for new transactions.

The model training and endpoint deployment is orchestrated by running a [jupyter notebook](source/notebooks/sagemaker_fraud_detection.ipynb) on a SageMaker Notebook instance. The jupyter notebook runs a demonstration of the project using the aforementioned anonymized credit card dataset that is automatically downloaded to the Amazon S3 Bucket created when you launch the solution. However, the notebook can be modified to run the project on a custom dataset in S3. The notebook instance also contains some example code that shows how to invoke the REST API for inference.

To make the project a standalone microservice, Amazon API Gateway is used to create a REST API, backed by an AWS Lambda function. The Lambda function handles the preprocessing of incoming transactions, invocation of SageMaker endpoints, merging of results, storage of model inputs and predictions in S3 via Kinesis Firehose, and providing a response to the client.



## Contents

* `deployment/`
  * `fraud-detection-using-machine-learning.yaml`: Creates AWS CloudFormation Stack for solution
* `source/`
  * `lambda`
    * `model-invocation/`
      * `index.py`: Lambda function script for invoking SageMaker endpoints for inference
  * `notebooks/`
    * `src`
      * `package`
        * `config.py`: Read in the environment variables set during the Amazon CloudFormation stack creation
        * `generate_endpoint_traffic.py`: Custom script to show how to send transaction traffic to REST API for inference
        * `util.py`: Helper function and utilities
    * `sagemaker_fraud_detection.ipynb`: Orchestrates the solution. Trains the models and deploys the trained model
    * `endpoint_demo.ipynb`: A small notebook that demonstrates how one can use the solution's endpoint to make prediction.
  * `scripts/`
    * `set_kernelspec.py`: Used to update the kernelspec name at deployment.
  * `test/`
    * Files that are used to automatically test the solution


## License

This project is licensed under the Apache-2.0 License.


