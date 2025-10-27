static void nfs_process_write(void *arg)

{

    NFSClient *client = arg;



    aio_context_acquire(client->aio_context);

    nfs_service(client->context, POLLOUT);

    nfs_set_events(client);

    aio_context_release(client->aio_context);

}
