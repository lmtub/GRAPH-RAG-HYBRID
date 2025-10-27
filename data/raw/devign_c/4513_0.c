static int udp_open(URLContext *h, const char *uri, int flags)

{

    char hostname[1024], localaddr[1024] = "";

    int port, udp_fd = -1, tmp, bind_ret = -1;

    UDPContext *s = h->priv_data;

    int is_output;

    const char *p;

    char buf[256];

    struct sockaddr_storage my_addr;

    int len;

    int reuse_specified = 0;



    h->is_streamed = 1;

    h->max_packet_size = 1472;



    is_output = !(flags & AVIO_FLAG_READ);



    s->ttl = 16;

    s->buffer_size = is_output ? UDP_TX_BUF_SIZE : UDP_MAX_PKT_SIZE;



    s->circular_buffer_size = 7*188*4096;



    p = strchr(uri, '?');

    if (p) {

        if (av_find_info_tag(buf, sizeof(buf), "reuse", p)) {

            char *endptr = NULL;

            s->reuse_socket = strtol(buf, &endptr, 10);

            /* assume if no digits were found it is a request to enable it */

            if (buf == endptr)

                s->reuse_socket = 1;

            reuse_specified = 1;

        }

        if (av_find_info_tag(buf, sizeof(buf), "overrun_nonfatal", p)) {

            char *endptr = NULL;

            s->overrun_nonfatal = strtol(buf, &endptr, 10);

            /* assume if no digits were found it is a request to enable it */

            if (buf == endptr)

                s->overrun_nonfatal = 1;

        }

        if (av_find_info_tag(buf, sizeof(buf), "ttl", p)) {

            s->ttl = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "localport", p)) {

            s->local_port = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "pkt_size", p)) {

            h->max_packet_size = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "buffer_size", p)) {

            s->buffer_size = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "connect", p)) {

            s->is_connected = strtol(buf, NULL, 10);

        }

        if (av_find_info_tag(buf, sizeof(buf), "fifo_size", p)) {

            s->circular_buffer_size = strtol(buf, NULL, 10)*188;

        }

        if (av_find_info_tag(buf, sizeof(buf), "localaddr", p)) {

            av_strlcpy(localaddr, buf, sizeof(localaddr));

        }

    }



    /* fill the dest addr */

    av_url_split(NULL, 0, NULL, 0, hostname, sizeof(hostname), &port, NULL, 0, uri);



    /* XXX: fix av_url_split */

    if (hostname[0] == '\0' || hostname[0] == '?') {

        /* only accepts null hostname if input */

        if (!(flags & AVIO_FLAG_READ))

            goto fail;

    } else {

        if (ff_udp_set_remote_url(h, uri) < 0)

            goto fail;

    }



    if ((s->is_multicast || !s->local_port) && (h->flags & AVIO_FLAG_READ))

        s->local_port = port;

    udp_fd = udp_socket_create(s, &my_addr, &len, localaddr);

    if (udp_fd < 0)

        goto fail;



    /* Follow the requested reuse option, unless it's multicast in which

     * case enable reuse unless explicitly disabled.

     */

    if (s->reuse_socket || (s->is_multicast && !reuse_specified)) {

        s->reuse_socket = 1;

        if (setsockopt (udp_fd, SOL_SOCKET, SO_REUSEADDR, &(s->reuse_socket), sizeof(s->reuse_socket)) != 0)

            goto fail;

    }



    /* If multicast, try binding the multicast address first, to avoid

     * receiving UDP packets from other sources aimed at the same UDP

     * port. This fails on windows. This makes sending to the same address

     * using sendto() fail, so only do it if we're opened in read-only mode. */

    if (s->is_multicast && !(h->flags & AVIO_FLAG_WRITE)) {

        bind_ret = bind(udp_fd,(struct sockaddr *)&s->dest_addr, len);

    }

    /* bind to the local address if not multicast or if the multicast

     * bind failed */

    /* the bind is needed to give a port to the socket now */

    if (bind_ret < 0 && bind(udp_fd,(struct sockaddr *)&my_addr, len) < 0) {

        av_log(h, AV_LOG_ERROR, "bind failed: %s\n", strerror(errno));

        goto fail;

    }



    len = sizeof(my_addr);

    getsockname(udp_fd, (struct sockaddr *)&my_addr, &len);

    s->local_port = udp_port(&my_addr, len);



    if (s->is_multicast) {

        if (h->flags & AVIO_FLAG_WRITE) {

            /* output */

            if (udp_set_multicast_ttl(udp_fd, s->ttl, (struct sockaddr *)&s->dest_addr) < 0)

                goto fail;

        }

        if (h->flags & AVIO_FLAG_READ) {

            /* input */

            if (udp_join_multicast_group(udp_fd, (struct sockaddr *)&s->dest_addr) < 0)

                goto fail;

        }

    }



    if (is_output) {

        /* limit the tx buf size to limit latency */

        tmp = s->buffer_size;

        if (setsockopt(udp_fd, SOL_SOCKET, SO_SNDBUF, &tmp, sizeof(tmp)) < 0) {

            av_log(h, AV_LOG_ERROR, "setsockopt(SO_SNDBUF): %s\n", strerror(errno));

            goto fail;

        }

    } else {

        /* set udp recv buffer size to the largest possible udp packet size to

         * avoid losing data on OSes that set this too low by default. */

        tmp = s->buffer_size;

        if (setsockopt(udp_fd, SOL_SOCKET, SO_RCVBUF, &tmp, sizeof(tmp)) < 0) {

            av_log(h, AV_LOG_WARNING, "setsockopt(SO_RECVBUF): %s\n", strerror(errno));

        }

        /* make the socket non-blocking */

        ff_socket_nonblock(udp_fd, 1);

    }

    if (s->is_connected) {

        if (connect(udp_fd, (struct sockaddr *) &s->dest_addr, s->dest_addr_len)) {

            av_log(h, AV_LOG_ERROR, "connect: %s\n", strerror(errno));

            goto fail;

        }

    }



    s->udp_fd = udp_fd;



#if HAVE_PTHREADS

    if (!is_output && s->circular_buffer_size) {

        int ret;



        /* start the task going */

        s->fifo = av_fifo_alloc(s->circular_buffer_size);

        ret = pthread_mutex_init(&s->mutex, NULL);

        if (ret != 0) {

            av_log(h, AV_LOG_ERROR, "pthread_mutex_init failed : %s\n", strerror(ret));

            goto fail;

        }

        ret = pthread_cond_init(&s->cond, NULL);

        if (ret != 0) {

            av_log(h, AV_LOG_ERROR, "pthread_cond_init failed : %s\n", strerror(ret));

            goto cond_fail;

        }

        ret = pthread_create(&s->circular_buffer_thread, NULL, circular_buffer_task, h);

        if (ret != 0) {

            av_log(h, AV_LOG_ERROR, "pthread_create failed : %s\n", strerror(ret));

            goto thread_fail;

        }

        s->thread_started = 1;

    }

#endif



    return 0;

#if HAVE_PTHREADS

 thread_fail:

    pthread_cond_destroy(&s->cond);

 cond_fail:

    pthread_mutex_destroy(&s->mutex);

#endif

 fail:

    if (udp_fd >= 0)

        closesocket(udp_fd);

    av_fifo_free(s->fifo);

    return AVERROR(EIO);

}
