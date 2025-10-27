static void nfs_file_close(BlockDriverState *bs)

{

    NFSClient *client = bs->opaque;

    nfs_client_close(client);

    qemu_mutex_destroy(&client->mutex);

}
