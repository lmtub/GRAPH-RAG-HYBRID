static void *iothread_run(void *opaque)

{

    IOThread *iothread = opaque;



    rcu_register_thread();



    my_iothread = iothread;

    qemu_mutex_lock(&iothread->init_done_lock);

    iothread->thread_id = qemu_get_thread_id();

    qemu_cond_signal(&iothread->init_done_cond);

    qemu_mutex_unlock(&iothread->init_done_lock);



    while (!atomic_read(&iothread->stopping)) {

        aio_poll(iothread->ctx, true);



        if (atomic_read(&iothread->worker_context)) {

            GMainLoop *loop;



            g_main_context_push_thread_default(iothread->worker_context);

            iothread->main_loop =

                g_main_loop_new(iothread->worker_context, TRUE);

            loop = iothread->main_loop;



            g_main_loop_run(iothread->main_loop);

            iothread->main_loop = NULL;

            g_main_loop_unref(loop);



            g_main_context_pop_thread_default(iothread->worker_context);

        }

    }



    rcu_unregister_thread();

    return NULL;

}
