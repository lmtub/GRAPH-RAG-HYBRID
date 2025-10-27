bool qio_task_propagate_error(QIOTask *task,

                              Error **errp)

{

    if (task->err) {

        error_propagate(errp, task->err);


        return true;

    }



    return false;

}