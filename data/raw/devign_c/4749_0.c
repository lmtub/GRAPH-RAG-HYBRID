static int tcp_open(URLContext *h, const char *uri, int flags)

{

    struct sockaddr_in dest_addr;

    char hostname[1024], *q;

    int port, fd = -1;

    TCPContext *s;

    const char *p;

    fd_set wfds;

    int fd_max, ret;

    struct timeval tv;

    socklen_t optlen;

    

    s = av_malloc(sizeof(TCPContext));

    if (!s)

        return -ENOMEM;

    h->priv_data = s;

    p = uri;

    if (!strstart(p, "tcp://", &p))

        goto fail;

    q = hostname;

    while (*p != ':' && *p != '/' && *p != '\0') {

        if ((q - hostname) < sizeof(hostname) - 1)

            *q++ = *p;

        p++;

    }

    *q = '\0';

    if (*p != ':')

        goto fail;

    p++;

    port = strtoul(p, (char **)&p, 10);

    if (port <= 0 || port >= 65536)

        goto fail;

    

    dest_addr.sin_family = AF_INET;

    dest_addr.sin_port = htons(port);

    if (resolve_host(&dest_addr.sin_addr, hostname) < 0)

        goto fail;



    fd = socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0)

        goto fail;

    fcntl(fd, F_SETFL, O_NONBLOCK);

    

 redo:

    ret = connect(fd, (struct sockaddr *)&dest_addr, 

                  sizeof(dest_addr));

    if (ret < 0) {

        if (errno == EINTR)

            goto redo;

        if (errno != EINPROGRESS)

            goto fail;



        /* wait until we are connected or until abort */

        for(;;) {

            if (url_interrupt_cb()) {

                ret = -EINTR;

                goto fail1;

            }

            fd_max = fd;

            FD_ZERO(&wfds);

            FD_SET(fd, &wfds);

            tv.tv_sec = 0;

            tv.tv_usec = 100 * 1000;

            ret = select(fd_max + 1, NULL, &wfds, NULL, &tv);

            if (ret > 0 && FD_ISSET(fd, &wfds))

                break;

        }

        

        /* test error */

        optlen = sizeof(ret);

        getsockopt (fd, SOL_SOCKET, SO_ERROR, &ret, &optlen);

        if (ret != 0)

            goto fail;

    }

    s->fd = fd;

    return 0;



 fail:

    ret = AVERROR_IO;

 fail1:

    if (fd >= 0)

        close(fd);

    av_free(s);

    return ret;

}
