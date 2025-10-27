static void iothread_instance_finalize(Object *obj)

{

    IOThread *iothread = IOTHREAD(obj);



    iothread_stop(obj, NULL);

    qemu_cond_destroy(&iothread->init_done_cond);

    qemu_mutex_destroy(&iothread->init_done_lock);




    aio_context_unref(iothread->ctx);
