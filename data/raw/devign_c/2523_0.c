static int spawn_thread(void)

{

    pthread_attr_t attr;

    int ret;



    cur_threads++;

    idle_threads++;



    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    ret = pthread_create(&thread_id, &attr, aio_thread, NULL);

    pthread_attr_destroy(&attr);



    return ret;

}
