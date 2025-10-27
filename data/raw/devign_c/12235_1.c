static int tcp_open(URLContext *h, const char *uri, int flags)

{

    struct sockaddr_in dest_addr;

    int port, fd = -1;

    TCPContext *s = NULL;

    fd_set wfds;

    int fd_max, ret;

    struct timeval tv;

    socklen_t optlen;

    char hostname[1024],proto[1024],path[1024],tmp[1024],*q;



    if(!ff_network_init())

        return AVERROR(EIO);



    url_split(proto, sizeof(proto), NULL, 0, hostname, sizeof(hostname),

        &port, path, sizeof(path), uri);

    if (strcmp(proto,"tcp") || port <= 0 || port >= 65536)

        return AVERROR(EINVAL);

    if ((q = strchr(hostname,'@'))) { strcpy(tmp,q+1); strcpy(hostname,tmp); }



    dest_addr.sin_family = AF_INET;

    dest_addr.sin_port = htons(port);

    if (resolve_host(&dest_addr.sin_addr, hostname) < 0)

        return AVERROR(EIO);



    fd = socket(AF_INET, SOCK_STREAM, 0);

    if (fd < 0)

        return AVERROR(EIO);

    ff_socket_nonblock(fd, 1);



 redo:

    ret = connect(fd, (struct sockaddr *)&dest_addr,

                  sizeof(dest_addr));

    if (ret < 0) {

        if (ff_neterrno() == FF_NETERROR(EINTR))

            goto redo;

        if (ff_neterrno() != FF_NETERROR(EINPROGRESS) &&

            ff_neterrno() != FF_NETERROR(EAGAIN))

            goto fail;



        /* wait until we are connected or until abort */

        for(;;) {

            if (url_interrupt_cb()) {

                ret = AVERROR(EINTR);

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

    s = av_malloc(sizeof(TCPContext));

    if (!s)

        return AVERROR(ENOMEM);

    h->priv_data = s;

    h->is_streamed = 1;

    s->fd = fd;

    return 0;



 fail:

    ret = AVERROR(EIO);

 fail1:

    if (fd >= 0)

        closesocket(fd);

    return ret;

}
