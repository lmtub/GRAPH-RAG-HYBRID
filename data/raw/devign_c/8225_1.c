static void new_connection(int server_fd, int is_rtsp)

{

    struct sockaddr_in from_addr;

    int fd, len;

    HTTPContext *c = NULL;



    len = sizeof(from_addr);

    fd = accept(server_fd, (struct sockaddr *)&from_addr, 

                &len);

    if (fd < 0)

        return;

    fcntl(fd, F_SETFL, O_NONBLOCK);



    /* XXX: should output a warning page when coming

       close to the connection limit */

    if (nb_connections >= nb_max_connections)

        goto fail;

    

    /* add a new connection */

    c = av_mallocz(sizeof(HTTPContext));

    if (!c)

        goto fail;

    

    c->next = first_http_ctx;

    first_http_ctx = c;

    c->fd = fd;

    c->poll_entry = NULL;

    c->from_addr = from_addr;

    c->buffer_size = IOBUFFER_INIT_SIZE;

    c->buffer = av_malloc(c->buffer_size);

    if (!c->buffer)

        goto fail;

    nb_connections++;

    

    start_wait_request(c, is_rtsp);



    return;



 fail:

    if (c) {

        av_free(c->buffer);

        av_free(c);

    }

    close(fd);

}
