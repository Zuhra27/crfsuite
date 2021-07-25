import logging

import brand_detection.crf
import brand_detection.data
import brand_detection.preprocess


logging.basicConfig(
    level=logging.INFO,
    style="{",
    format="{asctime}.{msecs:03.0f} {levelname:>7s} {process:d} "
    + "--- [{threadName:>15.15}] {name:<40.40s} : {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger(__name__)


def main():
    logger.info("Load Data...")
    product_data_frame = brand_detection.data.load_brand_data()
    logger.info("Assign Feartures...")
    data_with_features = brand_detection.preprocess.assign_feature_labels(
        product_data_frame
    )
    logger.info("Split data to train and test...")
    x_train, x_test, y_train, y_test = brand_detection.crf.train_test_split(
        data_with_features
    )

    params = {"c1": 0.05, "c2": 0.05, "max_iterations": 100}
    logger.info("Create trainer...")
    trainer = brand_detection.crf.create_trainer(params)
    logger.info("Fit model...")
    file = brand_detection.crf.fit(trainer, x_train, y_train)
    logger.info("Model file:  %s ", file.name)
    tagger = brand_detection.crf.create_tagger(file.name)
    logger.info("Predict...")
    predict = brand_detection.crf.predict(tagger, data_with_features["features"])
    logger.info("Report...")
    report = brand_detection.crf.get_classification_report(tagger, x_test, y_test)
    logger.info(report)
