static void fifo_deinit(AVFormatContext *avf)

{

    FifoContext *fifo = avf->priv_data;



    av_dict_free(&fifo->format_options);

    avformat_free_context(fifo->avf);

    av_thread_message_queue_free(&fifo->queue);

    pthread_mutex_destroy(&fifo->overflow_flag_lock);

}
