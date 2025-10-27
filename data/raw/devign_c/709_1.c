static int http_server(struct sockaddr_in my_addr)

{

    int server_fd, tmp, ret;

    struct sockaddr_in from_addr;

    struct pollfd poll_table[HTTP_MAX_CONNECTIONS + 1], *poll_entry;

    HTTPContext *c, **cp;

    long cur_time;



    server_fd = socket(AF_INET,SOCK_STREAM,0);

    if (server_fd < 0) {

        perror ("socket");

        return -1;

    }

        

    tmp = 1;

    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &tmp, sizeof(tmp));



    if (bind (server_fd, (struct sockaddr *) &my_addr, sizeof (my_addr)) < 0) {

        perror ("bind");

        close(server_fd);

        return -1;

    }

  

    if (listen (server_fd, 5) < 0) {

        perror ("listen");

        close(server_fd);

        return -1;

    }



    http_log("ffserver started.\n");



    start_children(first_feed);



    fcntl(server_fd, F_SETFL, O_NONBLOCK);

    first_http_ctx = NULL;

    nb_connections = 0;

    first_http_ctx = NULL;

    for(;;) {

        poll_entry = poll_table;

        poll_entry->fd = server_fd;

        poll_entry->events = POLLIN;

        poll_entry++;



        /* wait for events on each HTTP handle */

        c = first_http_ctx;

        while (c != NULL) {

            int fd;

            fd = c->fd;

            switch(c->state) {

            case HTTPSTATE_WAIT_REQUEST:

                c->poll_entry = poll_entry;

                poll_entry->fd = fd;

                poll_entry->events = POLLIN;

                poll_entry++;

                break;

            case HTTPSTATE_SEND_HEADER:

            case HTTPSTATE_SEND_DATA_HEADER:

            case HTTPSTATE_SEND_DATA:

            case HTTPSTATE_SEND_DATA_TRAILER:

                c->poll_entry = poll_entry;

                poll_entry->fd = fd;

                poll_entry->events = POLLOUT;

                poll_entry++;

                break;

            case HTTPSTATE_RECEIVE_DATA:

                c->poll_entry = poll_entry;

                poll_entry->fd = fd;

                poll_entry->events = POLLIN;

                poll_entry++;

                break;

            case HTTPSTATE_WAIT_FEED:

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

            ret = poll(poll_table, poll_entry - poll_table, 1000);

        } while (ret == -1);

        

        cur_time = gettime_ms();



        /* now handle the events */



        cp = &first_http_ctx;

        while ((*cp) != NULL) {

            c = *cp;

            if (handle_http (c, cur_time) < 0) {

                /* close and free the connection */

                log_connection(c);

                close(c->fd);

                if (c->fmt_in)

                    av_close_input_file(c->fmt_in);

                *cp = c->next;

                nb_bandwidth -= c->bandwidth;

                av_free(c->buffer);

                av_free(c->pbuffer);

                av_free(c);

                nb_connections--;

            } else {

                cp = &c->next;

            }

        }



        /* new connection request ? */

        poll_entry = poll_table;

        if (poll_entry->revents & POLLIN) {

            int fd, len;



            len = sizeof(from_addr);

            fd = accept(server_fd, (struct sockaddr *)&from_addr, 

                        &len);

            if (fd >= 0) {

                fcntl(fd, F_SETFL, O_NONBLOCK);

                /* XXX: should output a warning page when coming

                   close to the connection limit */

                if (nb_connections >= nb_max_connections) {

                    c = NULL;

                } else {

                    /* add a new connection */

                    c = av_mallocz(sizeof(HTTPContext));

                    if (c) {

                        c->next = first_http_ctx;

                        first_http_ctx = c;

                        c->fd = fd;

                        c->poll_entry = NULL;

                        c->from_addr = from_addr;

                        c->state = HTTPSTATE_WAIT_REQUEST;

                        c->buffer = av_malloc(c->buffer_size = IOBUFFER_INIT_SIZE);

                        c->pbuffer = av_malloc(c->pbuffer_size = PBUFFER_INIT_SIZE);

                        if (!c->buffer || !c->pbuffer) {

                            av_free(c->buffer);

                            av_free(c->pbuffer);

                            av_freep(&c);

                        } else {

                            c->buffer_ptr = c->buffer;

                            c->buffer_end = c->buffer + c->buffer_size;

                            c->timeout = cur_time + REQUEST_TIMEOUT;

                            c->start_time = cur_time;

                            nb_connections++;

                        }

                    }

                }

                if (!c) {

                    close(fd);

                }

            }

        }

        poll_entry++;

    }

}
