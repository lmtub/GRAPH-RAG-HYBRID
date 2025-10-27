static ssize_t qio_channel_websock_readv(QIOChannel *ioc,

                                         const struct iovec *iov,

                                         size_t niov,

                                         int **fds,

                                         size_t *nfds,

                                         Error **errp)

{

    QIOChannelWebsock *wioc = QIO_CHANNEL_WEBSOCK(ioc);

    size_t i;

    ssize_t got = 0;

    ssize_t ret;



    if (wioc->io_err) {

        *errp = error_copy(wioc->io_err);

        return -1;

    }



    if (!wioc->rawinput.offset) {

        ret = qio_channel_websock_read_wire(QIO_CHANNEL_WEBSOCK(ioc), errp);

        if (ret < 0) {

            return ret;

        }

    }



    for (i = 0 ; i < niov ; i++) {

        size_t want = iov[i].iov_len;

        if (want > (wioc->rawinput.offset - got)) {

            want = (wioc->rawinput.offset - got);

        }



        memcpy(iov[i].iov_base,

               wioc->rawinput.buffer + got,

               want);

        got += want;



        if (want < iov[i].iov_len) {

            break;

        }

    }



    buffer_advance(&wioc->rawinput, got);

    qio_channel_websock_set_watch(wioc);

    return got;

}
