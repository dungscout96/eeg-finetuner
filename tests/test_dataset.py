from eeg_finetuner.data import PreprocessingPipeline

pipeline = PreprocessingPipeline("ds005514")
data = pipeline.preprocess_dataset()