static int submit_packet(PerThreadContext *p, AVPacket *avpkt)

{

    FrameThreadContext *fctx = p->parent;

    PerThreadContext *prev_thread = fctx->prev_thread;

    const AVCodec *codec = p->avctx->codec;



    if (!avpkt->size && !(codec->capabilities & AV_CODEC_CAP_DELAY))

        return 0;



    pthread_mutex_lock(&p->mutex);



    release_delayed_buffers(p);



    if (prev_thread) {

        int err;

        if (prev_thread->state == STATE_SETTING_UP) {

            pthread_mutex_lock(&prev_thread->progress_mutex);

            while (prev_thread->state == STATE_SETTING_UP)

                pthread_cond_wait(&prev_thread->progress_cond, &prev_thread->progress_mutex);

            pthread_mutex_unlock(&prev_thread->progress_mutex);

        }



        err = update_context_from_thread(p->avctx, prev_thread->avctx, 0);

        if (err) {

            pthread_mutex_unlock(&p->mutex);

            return err;

        }

    }



    av_packet_unref(&p->avpkt);

    av_packet_ref(&p->avpkt, avpkt);



    p->state = STATE_SETTING_UP;

    pthread_cond_signal(&p->input_cond);

    pthread_mutex_unlock(&p->mutex);



    /*

     * If the client doesn't have a thread-safe get_buffer(),

     * then decoding threads call back to the main thread,

     * and it calls back to the client here.

     */



    if (!p->avctx->thread_safe_callbacks && (

         p->avctx->get_format != avcodec_default_get_format ||

         p->avctx->get_buffer2 != avcodec_default_get_buffer2)) {

        while (p->state != STATE_SETUP_FINISHED && p->state != STATE_INPUT_READY) {

            int call_done = 1;

            pthread_mutex_lock(&p->progress_mutex);

            while (p->state == STATE_SETTING_UP)

                pthread_cond_wait(&p->progress_cond, &p->progress_mutex);



            switch (p->state) {

            case STATE_GET_BUFFER:

                p->result = ff_get_buffer(p->avctx, p->requested_frame, p->requested_flags);

                break;

            case STATE_GET_FORMAT:

                p->result_format = ff_get_format(p->avctx, p->available_formats);

                break;

            default:

                call_done = 0;

                break;

            }

            if (call_done) {

                p->state  = STATE_SETTING_UP;

                pthread_cond_signal(&p->progress_cond);

            }

            pthread_mutex_unlock(&p->progress_mutex);

        }

    }



    fctx->prev_thread = p;

    fctx->next_decoding++;



    return 0;

}
