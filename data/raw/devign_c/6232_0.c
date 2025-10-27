void qio_channel_socket_dgram_async(QIOChannelSocket *ioc,

                                    SocketAddress *localAddr,

                                    SocketAddress *remoteAddr,

                                    QIOTaskFunc callback,

                                    gpointer opaque,

                                    GDestroyNotify destroy)

{

    QIOTask *task = qio_task_new(

        OBJECT(ioc), callback, opaque, destroy);

    struct QIOChannelSocketDGramWorkerData *data = g_new0(

        struct QIOChannelSocketDGramWorkerData, 1);



    data->localAddr = QAPI_CLONE(SocketAddress, localAddr);

    data->remoteAddr = QAPI_CLONE(SocketAddress, remoteAddr);



    trace_qio_channel_socket_dgram_async(ioc, localAddr, remoteAddr);

    qio_task_run_in_thread(task,

                           qio_channel_socket_dgram_worker,

                           data,

                           qio_channel_socket_dgram_worker_free);

}
