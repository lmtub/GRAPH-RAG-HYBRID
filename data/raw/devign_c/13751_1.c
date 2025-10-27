void av_thread_message_queue_set_err_send(AVThreadMessageQueue *mq,

                                          int err)

{

#if HAVE_THREADS

    pthread_mutex_lock(&mq->lock);

    mq->err_send = err;

    pthread_cond_broadcast(&mq->cond);

    pthread_mutex_unlock(&mq->lock);

#endif /* HAVE_THREADS */

}
