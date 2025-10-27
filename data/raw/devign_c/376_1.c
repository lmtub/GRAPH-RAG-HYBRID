static int http_receive_data(HTTPContext *c)

{

    int len;

    HTTPContext *c1;



    if (c->buffer_ptr >= c->buffer_end) {

        FFStream *feed = c->stream;

        /* a packet has been received : write it in the store, except

           if header */

        if (c->data_count > FFM_PACKET_SIZE) {

            

            //            printf("writing pos=0x%Lx size=0x%Lx\n", feed->feed_write_index, feed->feed_size);

            /* XXX: use llseek or url_seek */

            lseek(c->feed_fd, feed->feed_write_index, SEEK_SET);

            write(c->feed_fd, c->buffer, FFM_PACKET_SIZE);

            

            feed->feed_write_index += FFM_PACKET_SIZE;

            /* update file size */

            if (feed->feed_write_index > c->stream->feed_size)

                feed->feed_size = feed->feed_write_index;



            /* handle wrap around if max file size reached */

            if (feed->feed_write_index >= c->stream->feed_max_size)

                feed->feed_write_index = FFM_PACKET_SIZE;



            /* write index */

            ffm_write_write_index(c->feed_fd, feed->feed_write_index);



            /* wake up any waiting connections */

            for(c1 = first_http_ctx; c1 != NULL; c1 = c1->next) {

                if (c1->state == HTTPSTATE_WAIT_FEED && 

                    c1->stream->feed == c->stream->feed) {

                    c1->state = HTTPSTATE_SEND_DATA;

                }

            }

        } else {

            /* We have a header in our hands that contains useful data */

            AVFormatContext s;

            ByteIOContext *pb = &s.pb;

            int i;



            memset(&s, 0, sizeof(s));



            url_open_buf(pb, c->buffer, c->buffer_end - c->buffer, URL_RDONLY);

            pb->buf_end = c->buffer_end;        /* ?? */

            pb->is_streamed = 1;



            if (feed->fmt->read_header(&s, 0) < 0) {

                goto fail;

            }



            /* Now we have the actual streams */

            if (s.nb_streams != feed->nb_streams) {

                goto fail;

            }

            for (i = 0; i < s.nb_streams; i++) {

                memcpy(&feed->streams[i]->codec, &s.streams[i]->codec, sizeof(AVCodecContext));

            } 

        }

        c->buffer_ptr = c->buffer;

    }



    len = read(c->fd, c->buffer_ptr, c->buffer_end - c->buffer_ptr);

    if (len < 0) {

        if (errno != EAGAIN && errno != EINTR) {

            /* error : close connection */

            goto fail;

        }

    } else if (len == 0) {

        /* end of connection : close it */

        goto fail;

    } else {

        c->buffer_ptr += len;

        c->data_count += len;

    }

    return 0;

 fail:

    c->stream->feed_opened = 0;

    close(c->feed_fd);

    return -1;

}
