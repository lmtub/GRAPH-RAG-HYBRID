static int qio_channel_socket_dgram_worker(QIOTask *task,

                                           Error **errp,

                                           gpointer opaque)

{

    QIOChannelSocket *ioc = QIO_CHANNEL_SOCKET(qio_task_get_source(task));

    struct QIOChannelSocketDGramWorkerData *data = opaque;

    int ret;



    /* socket_dgram() blocks in DNS lookups, so we must use a thread */

    ret = qio_channel_socket_dgram_sync(ioc,

                                        data->localAddr,

                                        data->remoteAddr,

                                        errp);



    object_unref(OBJECT(ioc));

    return ret;

}
