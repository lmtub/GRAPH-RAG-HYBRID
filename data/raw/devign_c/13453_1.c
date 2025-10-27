static int qio_channel_command_close(QIOChannel *ioc,

                                     Error **errp)

{

    QIOChannelCommand *cioc = QIO_CHANNEL_COMMAND(ioc);

    int rv = 0;



    /* We close FDs before killing, because that

     * gives a better chance of clean shutdown

     */

    if (close(cioc->writefd) < 0) {

        rv = -1;

    }

    if (close(cioc->readfd) < 0) {

        rv = -1;

    }

#ifndef WIN32

    if (qio_channel_command_abort(cioc, errp) < 0) {

        return -1;

    }

#endif

    if (rv < 0) {

        error_setg_errno(errp, errno, "%s",

                         "Unable to close command");

    }

    return rv;

}
