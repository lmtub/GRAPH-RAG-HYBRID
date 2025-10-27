static int handle_connection(HTTPContext *c)

{

    int len, ret;

    

    switch(c->state) {

    case HTTPSTATE_WAIT_REQUEST:

    case RTSPSTATE_WAIT_REQUEST:

        /* timeout ? */

        if ((c->timeout - cur_time) < 0)

            return -1;

        if (c->poll_entry->revents & (POLLERR | POLLHUP))

            return -1;



        /* no need to read if no events */

        if (!(c->poll_entry->revents & POLLIN))

            return 0;

        /* read the data */

        len = read(c->fd, c->buffer_ptr, 1);

        if (len < 0) {

            if (errno != EAGAIN && errno != EINTR)

                return -1;

        } else if (len == 0) {

            return -1;

        } else {

            /* search for end of request. */

            uint8_t *ptr;

            c->buffer_ptr += len;

            ptr = c->buffer_ptr;

            if ((ptr >= c->buffer + 2 && !memcmp(ptr-2, "\n\n", 2)) ||

                (ptr >= c->buffer + 4 && !memcmp(ptr-4, "\r\n\r\n", 4))) {

                /* request found : parse it and reply */

                if (c->state == HTTPSTATE_WAIT_REQUEST) {

                    ret = http_parse_request(c);

                } else {

                    ret = rtsp_parse_request(c);

                }

                if (ret < 0)

                    return -1;

            } else if (ptr >= c->buffer_end) {

                /* request too long: cannot do anything */

                return -1;

            }

        }

        break;



    case HTTPSTATE_SEND_HEADER:

        if (c->poll_entry->revents & (POLLERR | POLLHUP))

            return -1;



        /* no need to write if no events */

        if (!(c->poll_entry->revents & POLLOUT))

            return 0;

        len = write(c->fd, c->buffer_ptr, c->buffer_end - c->buffer_ptr);

        if (len < 0) {

            if (errno != EAGAIN && errno != EINTR) {

                /* error : close connection */

                av_freep(&c->pb_buffer);

                return -1;

            }

        } else {

            c->buffer_ptr += len;

            if (c->stream)

                c->stream->bytes_served += len;

            c->data_count += len;

            if (c->buffer_ptr >= c->buffer_end) {

                av_freep(&c->pb_buffer);

                /* if error, exit */

                if (c->http_error) {

                    return -1;

                }

                /* all the buffer was sent : synchronize to the incoming stream */

                c->state = HTTPSTATE_SEND_DATA_HEADER;

                c->buffer_ptr = c->buffer_end = c->buffer;

            }

        }

        break;



    case HTTPSTATE_SEND_DATA:

    case HTTPSTATE_SEND_DATA_HEADER:

    case HTTPSTATE_SEND_DATA_TRAILER:

        /* for packetized output, we consider we can always write (the

           input streams sets the speed). It may be better to verify

           that we do not rely too much on the kernel queues */

        if (!c->is_packetized) {

            if (c->poll_entry->revents & (POLLERR | POLLHUP))

                return -1;

            

            /* no need to read if no events */

            if (!(c->poll_entry->revents & POLLOUT))

                return 0;

        }

        if (http_send_data(c) < 0)

            return -1;

        break;

    case HTTPSTATE_RECEIVE_DATA:

        /* no need to read if no events */

        if (c->poll_entry->revents & (POLLERR | POLLHUP))

            return -1;

        if (!(c->poll_entry->revents & POLLIN))

            return 0;

        if (http_receive_data(c) < 0)

            return -1;

        break;

    case HTTPSTATE_WAIT_FEED:

        /* no need to read if no events */

        if (c->poll_entry->revents & (POLLIN | POLLERR | POLLHUP))

            return -1;



        /* nothing to do, we'll be waken up by incoming feed packets */

        break;



    case HTTPSTATE_WAIT:

        /* if the delay expired, we can send new packets */

        if (compute_send_delay(c) <= 0)

            c->state = HTTPSTATE_SEND_DATA;

        break;

    case HTTPSTATE_WAIT_SHORT:

        /* just return back to send data */

        c->state = HTTPSTATE_SEND_DATA;

        break;



    case RTSPSTATE_SEND_REPLY:

        if (c->poll_entry->revents & (POLLERR | POLLHUP)) {

            av_freep(&c->pb_buffer);

            return -1;

        }

        /* no need to write if no events */

        if (!(c->poll_entry->revents & POLLOUT))

            return 0;

        len = write(c->fd, c->buffer_ptr, c->buffer_end - c->buffer_ptr);

        if (len < 0) {

            if (errno != EAGAIN && errno != EINTR) {

                /* error : close connection */

                av_freep(&c->pb_buffer);

                return -1;

            }

        } else {

            c->buffer_ptr += len;

            c->data_count += len;

            if (c->buffer_ptr >= c->buffer_end) {

                /* all the buffer was sent : wait for a new request */

                av_freep(&c->pb_buffer);

                start_wait_request(c, 1);

            }

        }

        break;

    case HTTPSTATE_READY:

        /* nothing to do */

        break;

    default:

        return -1;

    }

    return 0;

}
