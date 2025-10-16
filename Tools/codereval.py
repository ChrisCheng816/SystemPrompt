from datasets import load_dataset
from datasets import DatasetDict

ds_java_train = load_dataset("google/code_x_glue_tc_text_to_code")["train"]
ds_java_test = load_dataset("vitaleantonio/codereval-java")["train"]

ds_java_train = ds_java_train.rename_column("nl", "instruction")
ds_java_train = ds_java_train.rename_column("code", "output")

ds_java_test = ds_java_test.rename_column("input", "instruction")

codereval_java = DatasetDict({
    "train": ds_java_train,
    "test": ds_java_test
})


ds_python_train = load_dataset("google/code_x_glue_tc_nl_code_search_adv")["train"]
ds_python_train = ds_python_train.rename_column("docstring", "instruction")
ds_python_train = ds_python_train.rename_column("code", "output")

ds_python_test = load_dataset("vitaleantonio/codereval-python")["train"]
ds_python_test = ds_python_test.rename_column("input", "instruction")

codereval_python = DatasetDict({
    "train": ds_python_train,
    "test": ds_python_test
})
print(codereval_python)
print(codereval_java)