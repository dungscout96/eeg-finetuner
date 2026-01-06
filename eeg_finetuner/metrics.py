from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

binary_classification_metric_collection = MetricCollection([
    MulticlassAccuracy(num_classes=2, average="micro"),
    MulticlassPrecision(num_classes=2, average="macro"),
    MulticlassRecall(num_classes=2, average="macro")
])

multiclass_classification_metric_collection = MetricCollection([
    MulticlassAccuracy(num_classes=3, average="micro"),
    MulticlassPrecision(num_classes=3, average="macro"),
    MulticlassRecall(num_classes=3, average="macro")
])