from clearml import Task
import os

def load_model_from_clearml():
    task_id = os.getenv("TRAINING_TASK_ID")
    if not task_id:
        raise RuntimeError("TRAINING_TASK_ID not set")

    task = Task.get_task(task_id=task_id)
    model = task.models["output"][0]
    return model.get_local_copy()
