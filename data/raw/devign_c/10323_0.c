static void qio_channel_websock_finalize(Object *obj)

{

    QIOChannelWebsock *ioc = QIO_CHANNEL_WEBSOCK(obj);



    buffer_free(&ioc->encinput);

    buffer_free(&ioc->encoutput);

    buffer_free(&ioc->rawinput);

    buffer_free(&ioc->rawoutput);

    object_unref(OBJECT(ioc->master));

    if (ioc->io_tag) {

        g_source_remove(ioc->io_tag);

    }

    if (ioc->io_err) {

        error_free(ioc->io_err);

    }

}
