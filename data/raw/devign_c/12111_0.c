int qemu_paio_init(struct qemu_paioinit *aioinit)

{

    int ret;



    ret = pthread_attr_init(&attr);

    if (ret) die2(ret, "pthread_attr_init");



    ret = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    if (ret) die2(ret, "pthread_attr_setdetachstate");



    TAILQ_INIT(&request_list);



    return 0;

}
