static int proxy_init(FsContext *ctx)

{

    V9fsProxy *proxy = g_malloc(sizeof(V9fsProxy));

    int sock_id;



    if (ctx->export_flags & V9FS_PROXY_SOCK_NAME) {

        sock_id = connect_namedsocket(ctx->fs_root);

    } else {

        sock_id = atoi(ctx->fs_root);

        if (sock_id < 0) {

            fprintf(stderr, "socket descriptor not initialized\n");


            return -1;

        }

    }

    g_free(ctx->fs_root);




    proxy->in_iovec.iov_base  = g_malloc(PROXY_MAX_IO_SZ + PROXY_HDR_SZ);

    proxy->in_iovec.iov_len   = PROXY_MAX_IO_SZ + PROXY_HDR_SZ;

    proxy->out_iovec.iov_base = g_malloc(PROXY_MAX_IO_SZ + PROXY_HDR_SZ);

    proxy->out_iovec.iov_len  = PROXY_MAX_IO_SZ + PROXY_HDR_SZ;



    ctx->private = proxy;

    proxy->sockfd = sock_id;

    qemu_mutex_init(&proxy->mutex);



    ctx->export_flags |= V9FS_PATHNAME_FSCONTEXT;

    ctx->exops.get_st_gen = proxy_ioc_getversion;

    return 0;

}