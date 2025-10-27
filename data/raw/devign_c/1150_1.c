static gboolean qio_channel_websock_handshake_send(QIOChannel *ioc,

                                                   GIOCondition condition,

                                                   gpointer user_data)

{

    QIOTask *task = user_data;

    QIOChannelWebsock *wioc = QIO_CHANNEL_WEBSOCK(

        qio_task_get_source(task));

    Error *err = NULL;

    ssize_t ret;



    ret = qio_channel_write(wioc->master,

                            (char *)wioc->encoutput.buffer,

                            wioc->encoutput.offset,

                            &err);



    if (ret < 0) {

        trace_qio_channel_websock_handshake_fail(ioc);

        qio_task_set_error(task, err);

        qio_task_complete(task);

        return FALSE;

    }



    buffer_advance(&wioc->encoutput, ret);

    if (wioc->encoutput.offset == 0) {

        trace_qio_channel_websock_handshake_complete(ioc);

        qio_task_complete(task);

        return FALSE;

    }

    trace_qio_channel_websock_handshake_pending(ioc, G_IO_OUT);

    return TRUE;

}
