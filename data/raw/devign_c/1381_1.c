bool qemu_co_enter_next(CoQueue *queue)

{

    Coroutine *next;



    next = QSIMPLEQ_FIRST(&queue->entries);

    if (!next) {

        return false;

    }



    QSIMPLEQ_REMOVE_HEAD(&queue->entries, co_queue_next);

    qemu_coroutine_enter(next, NULL);

    return true;

}
