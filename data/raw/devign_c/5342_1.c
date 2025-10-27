static gboolean qio_channel_websock_handshake_io(QIOChannel *ioc,

                                                 GIOCondition condition,

                                                 gpointer user_data)

{

    QIOTask *task = user_data;

    QIOChannelWebsock *wioc = QIO_CHANNEL_WEBSOCK(

        qio_task_get_source(task));

    Error *err = NULL;

    int ret;



    ret = qio_channel_websock_handshake_read(wioc, &err);

    if (ret < 0) {

        trace_qio_channel_websock_handshake_fail(ioc);

        qio_task_abort(task, err);

        error_free(err);

        return FALSE;

    }

    if (ret == 0) {

        trace_qio_channel_websock_handshake_pending(ioc, G_IO_IN);

        /* need more data still */

        return TRUE;

    }



    object_ref(OBJECT(task));

    trace_qio_channel_websock_handshake_reply(ioc);

    qio_channel_add_watch(

        wioc->master,

        G_IO_OUT,

        qio_channel_websock_handshake_send,

        task,

        (GDestroyNotify)object_unref);

    return FALSE;

}
