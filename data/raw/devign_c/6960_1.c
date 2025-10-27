static void qemu_rbd_close(BlockDriverState *bs)

{

    BDRVRBDState *s = bs->opaque;



    close(s->fds[0]);

    close(s->fds[1]);

    qemu_aio_set_fd_handler(s->fds[RBD_FD_READ], NULL, NULL, NULL);



    rbd_close(s->image);

    rados_ioctx_destroy(s->io_ctx);

    g_free(s->snap);

    rados_shutdown(s->cluster);

}
