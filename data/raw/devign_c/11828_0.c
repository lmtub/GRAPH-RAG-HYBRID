static int floppy_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVRawState *s = bs->opaque;

    int ret;



    posix_aio_init();



    s->type = FTYPE_FD;



    /* open will not fail even if no floppy is inserted, so add O_NONBLOCK */

    ret = raw_open_common(bs, filename, flags, O_NONBLOCK);

    if (ret)

        return ret;



    /* close fd so that we can reopen it as needed */

    close(s->fd);

    s->fd = -1;

    s->fd_media_changed = 1;



    return 0;

}
