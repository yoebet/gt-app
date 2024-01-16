def get_task_dir(tasks_dir: str, task_id: str, sub_dir: str = None):
    if sub_dir == '' or sub_dir == '_':
        sub_dir = None
    if sub_dir is not None:
        task_dir = f'{tasks_dir}/{sub_dir}/t_{task_id}'
    else:
        task_dir = f'{tasks_dir}/t_{task_id}'

    return task_dir


def get_tf_logging_dir(tf_logs_dir: str, task_id: str, sub_dir: str = None):
    if sub_dir == '' or sub_dir == '_':
        sub_dir = None
    if sub_dir is not None:
        logging_dir = f'{tf_logs_dir}/{sub_dir}/t_{task_id}'
    else:
        logging_dir = f'{tf_logs_dir}/t_{task_id}'
    return logging_dir
