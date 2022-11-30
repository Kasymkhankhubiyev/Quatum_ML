
class DoesntMatchChosenTask(Exception):
    """Chosen task does not match """
    def __init__(self, tasks_list, err_task):
        super().__init__(
            f'The shosem task {err_task} does not'
            f'match with available {tasks_list}'
        )
