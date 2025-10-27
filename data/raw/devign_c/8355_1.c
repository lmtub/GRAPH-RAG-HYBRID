void ff_frame_thread_encoder_free(AVCodecContext *avctx){

    int i;

    ThreadContext *c= avctx->internal->frame_thread_encoder;



    pthread_mutex_lock(&c->task_fifo_mutex);

    c->exit = 1;

    pthread_cond_broadcast(&c->task_fifo_cond);

    pthread_mutex_unlock(&c->task_fifo_mutex);



    for (i=0; i<avctx->thread_count; i++) {

         pthread_join(c->worker[i], NULL);

    }



    pthread_mutex_destroy(&c->task_fifo_mutex);

    pthread_mutex_destroy(&c->finished_task_mutex);

    pthread_mutex_destroy(&c->buffer_mutex);

    pthread_cond_destroy(&c->task_fifo_cond);

    pthread_cond_destroy(&c->finished_task_cond);

    av_fifo_freep(&c->task_fifo);

    av_freep(&avctx->internal->frame_thread_encoder);

}
