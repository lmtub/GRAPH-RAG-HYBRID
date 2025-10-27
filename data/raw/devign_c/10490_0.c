static int udp_read(URLContext *h, uint8_t *buf, int size)

{

    UDPContext *s = h->priv_data;

    int ret;

    int avail;



#if HAVE_PTHREADS

    if (s->fifo) {

        pthread_mutex_lock(&s->mutex);

        do {

            avail = av_fifo_size(s->fifo);

            if (avail) { // >=size) {

                uint8_t tmp[4];

                pthread_mutex_unlock(&s->mutex);



                av_fifo_generic_read(s->fifo, tmp, 4, NULL);

                avail= AV_RL32(tmp);

                if(avail > size){

                    av_log(h, AV_LOG_WARNING, "Part of datagram lost due to insufficient buffer size\n");

                    avail= size;

                }



                av_fifo_generic_read(s->fifo, buf, avail, NULL);

                av_fifo_drain(s->fifo, AV_RL32(tmp) - avail);

                return avail;

            } else if(s->circular_buffer_error){

                pthread_mutex_unlock(&s->mutex);

                return s->circular_buffer_error;

            } else if(h->flags & AVIO_FLAG_NONBLOCK) {

                pthread_mutex_unlock(&s->mutex);

                return AVERROR(EAGAIN);

            }

            else {

                pthread_cond_wait(&s->cond, &s->mutex);

            }

        } while( 1);

    }

#endif



    if (!(h->flags & AVIO_FLAG_NONBLOCK)) {

        ret = ff_network_wait_fd(s->udp_fd, 0);

        if (ret < 0)

            return ret;

    }

    ret = recv(s->udp_fd, buf, size, 0);



    return ret < 0 ? ff_neterrno() : ret;

}
