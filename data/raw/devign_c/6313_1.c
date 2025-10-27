static int unix_open(URLContext *h, const char *filename, int flags)

{

    UnixContext *s = h->priv_data;

    int fd, ret;



    av_strstart(filename, "unix:", &filename);

    s->addr.sun_family = AF_UNIX;

    av_strlcpy(s->addr.sun_path, filename, sizeof(s->addr.sun_path));



    if ((fd = ff_socket(AF_UNIX, s->type, 0)) < 0)

        return ff_neterrno();



    if (s->listen) {

        fd = ff_listen_bind(fd, (struct sockaddr *)&s->addr,

                            sizeof(s->addr), s->timeout, h);

        if (fd < 0) {

            ret = fd;

            goto fail;

        }

    } else {

        ret = ff_listen_connect(fd, (struct sockaddr *)&s->addr,

                                sizeof(s->addr), s->timeout, h, 0);

        if (ret < 0)

            goto fail;

    }



    s->fd = fd;



    return 0;



fail:

    if (s->listen && AVUNERROR(ret) != EADDRINUSE)

        unlink(s->addr.sun_path);

    if (fd >= 0)

        closesocket(fd);

    return ret;

}
