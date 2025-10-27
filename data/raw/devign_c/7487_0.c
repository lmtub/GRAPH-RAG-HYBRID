static void do_spawn_thread(ThreadPool *pool)

{

    QemuThread t;



    /* Runs with lock taken.  */

    if (!pool->new_threads) {

        return;

    }



    pool->new_threads--;

    pool->pending_threads++;



    qemu_thread_create(&t, "worker", worker_thread, pool, QEMU_THREAD_DETACHED);

}
