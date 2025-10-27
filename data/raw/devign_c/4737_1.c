static int http_send_data(HTTPContext *c, long cur_time)

{

    int len, ret;



    while (c->buffer_ptr >= c->buffer_end) {

        ret = http_prepare_data(c, cur_time);

        if (ret < 0)

            return -1;

        else if (ret == 0) {

            continue;

        } else {

            /* state change requested */

            return 0;

        }

    }



    if (c->buffer_end > c->buffer_ptr) {

        len = write(c->fd, c->buffer_ptr, c->buffer_end - c->buffer_ptr);

        if (len < 0) {

            if (errno != EAGAIN && errno != EINTR) {

                /* error : close connection */

                return -1;

            }

        } else {

            c->buffer_ptr += len;

            c->data_count += len;

            if (c->stream)

                c->stream->bytes_served += len;

        }

    }

    return 0;

}
