static void *circular_buffer_task( void *_URLContext)

{

    URLContext *h = _URLContext;

    UDPContext *s = h->priv_data;

    int old_cancelstate;



    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_cancelstate);

    ff_socket_nonblock(s->udp_fd, 0);

    while(1) {

        int left;

        int len;



        /* How much do we have left to the end of the buffer */

        /* Whats the minimum we can read so that we dont comletely fill the buffer */

        left = av_fifo_space(s->fifo);



        /* Blocking operations are always cancellation points;

           see "General Information" / "Thread Cancelation Overview"

           in Single Unix. */

        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_cancelstate);

        len = recv(s->udp_fd, s->tmp+4, sizeof(s->tmp)-4, 0);

        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &old_cancelstate);

        if (len < 0) {

            if (ff_neterrno() != AVERROR(EAGAIN) && ff_neterrno() != AVERROR(EINTR)) {

                s->circular_buffer_error = AVERROR(EIO);

                goto end;

            }

            continue;

        }

        AV_WL32(s->tmp, len);

        if(left < len + 4) {

            /* No Space left */

            if (s->overrun_nonfatal) {

                av_log(h, AV_LOG_WARNING, "Circular buffer overrun. "

                        "Surviving due to overrun_nonfatal option\n");

                continue;

            } else {

                av_log(h, AV_LOG_ERROR, "Circular buffer overrun. "

                        "To avoid, increase fifo_size URL option. "

                        "To survive in such case, use overrun_nonfatal option\n");

                s->circular_buffer_error = AVERROR(EIO);

                goto end;

            }

        }

        pthread_mutex_lock(&s->mutex);

        av_fifo_generic_write(s->fifo, s->tmp, len+4, NULL);

        pthread_cond_signal(&s->cond);

        pthread_mutex_unlock(&s->mutex);

    }



end:

    pthread_mutex_lock(&s->mutex);

    pthread_cond_signal(&s->cond);

    pthread_mutex_unlock(&s->mutex);

    return NULL;

}
