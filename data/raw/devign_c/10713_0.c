static void qio_channel_socket_dgram_worker_free(gpointer opaque)

{

    struct QIOChannelSocketDGramWorkerData *data = opaque;

    qapi_free_SocketAddressLegacy(data->localAddr);

    qapi_free_SocketAddressLegacy(data->remoteAddr);

    g_free(data);

}
