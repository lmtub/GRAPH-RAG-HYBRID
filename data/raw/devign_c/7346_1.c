static void qemu_co_queue_next_bh(void *opaque)

{

    struct unlock_bh *unlock_bh = opaque;

    Coroutine *next;



    trace_qemu_co_queue_next_bh();

    while ((next = QTAILQ_FIRST(&unlock_bh_queue))) {

        QTAILQ_REMOVE(&unlock_bh_queue, next, co_queue_next);

        qemu_coroutine_enter(next, NULL);

    }



    qemu_bh_delete(unlock_bh->bh);

    qemu_free(unlock_bh);

}
