static int http_server(void)

{

    int server_fd, ret, rtsp_server_fd, delay, delay1;

    struct pollfd poll_table[HTTP_MAX_CONNECTIONS + 2], *poll_entry;

    HTTPContext *c, *c_next;



    server_fd = socket_open_listen(&my_http_addr);

    if (server_fd < 0)

        return -1;



    rtsp_server_fd = socket_open_listen(&my_rtsp_addr);

    if (rtsp_server_fd < 0)

        return -1;



    http_log("ffserver started.\n");



    start_children(first_feed);



    first_http_ctx = NULL;

    nb_connections = 0;



    start_multicast();



    for(;;) {

        poll_entry = poll_table;

        poll_entry->fd = server_fd;

        poll_entry->events = POLLIN;

        poll_entry++;



        poll_entry->fd = rtsp_server_fd;

        poll_entry->events = POLLIN;

        poll_entry++;



        /* wait for events on each HTTP handle */

        c = first_http_ctx;

        delay = 1000;

        while (c != NULL) {

            int fd;

            fd = c->fd;

            switch(c->state) {

            case HTTPSTATE_SEND_HEADER:

            case RTSPSTATE_SEND_REPLY:

            case RTSPSTATE_SEND_PACKET:

                c->poll_entry = poll_entry;

                poll_entry->fd = fd;

                poll_entry->events = POLLOUT;

                poll_entry++;

                break;

            case HTTPSTATE_SEND_DATA_HEADER:

            case HTTPSTATE_SEND_DATA:

            case HTTPSTATE_SEND_DATA_TRAILER:

                if (!c->is_packetized) {

                    /* for TCP, we output as much as we can (may need to put a limit) */

                    c->poll_entry = poll_entry;

                    poll_entry->fd = fd;

                    poll_entry->events = POLLOUT;

                    poll_entry++;

                } else {

                    /* when ffserver is doing the timing, we work by

                       looking at which packet need to be sent every

                       10 ms */

                    delay1 = 10; /* one tick wait XXX: 10 ms assumed */

                    if (delay1 < delay)

                        delay = delay1;

                }

                break;

            case HTTPSTATE_WAIT_REQUEST:

            case HTTPSTATE_RECEIVE_DATA:

            case HTTPSTATE_WAIT_FEED:

            case RTSPSTATE_WAIT_REQUEST:

                /* need to catch errors */

                c->poll_entry = poll_entry;

                poll_entry->fd = fd;

                poll_entry->events = POLLIN;/* Maybe this will work */

                poll_entry++;

                break;

            default:

                c->poll_entry = NULL;

                break;

            }

            c = c->next;

        }



        /* wait for an event on one connection. We poll at least every

           second to handle timeouts */

        do {

            ret = poll(poll_table, poll_entry - poll_table, delay);

            if (ret < 0 && errno != EAGAIN && errno != EINTR)

                return -1;

        } while (ret <= 0);



        cur_time = av_gettime() / 1000;



        if (need_to_start_children) {

            need_to_start_children = 0;

            start_children(first_feed);

        }



        /* now handle the events */

        for(c = first_http_ctx; c != NULL; c = c_next) {

            c_next = c->next;

            if (handle_connection(c) < 0) {

                /* close and free the connection */

                log_connection(c);

                close_connection(c);

            }

        }



        poll_entry = poll_table;

        /* new HTTP connection request ? */

        if (poll_entry->revents & POLLIN) {

            new_connection(server_fd, 0);

        }

        poll_entry++;

        /* new RTSP connection request ? */

        if (poll_entry->revents & POLLIN) {

            new_connection(rtsp_server_fd, 1);

        }

    }

}
