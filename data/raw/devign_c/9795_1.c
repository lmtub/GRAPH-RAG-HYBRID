static void raw_close(BlockDriverState *bs)

{

    BDRVRawState *s = bs->opaque;

    if (s->fd >= 0) {

        close(s->fd);

        s->fd = -1;

        if (s->aligned_buf != NULL)

            qemu_free(s->aligned_buf);

    }

}
