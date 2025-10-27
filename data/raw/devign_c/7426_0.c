static void raw_refresh_limits(BlockDriverState *bs, Error **errp)

{

    BDRVRawState *s = bs->opaque;

    struct stat st;



    if (!fstat(s->fd, &st)) {

        if (S_ISBLK(st.st_mode)) {

            int ret = hdev_get_max_transfer_length(s->fd);

            if (ret >= 0) {

                bs->bl.max_transfer_length = ret;

            }

        }

    }



    raw_probe_alignment(bs, s->fd, errp);

    bs->bl.min_mem_alignment = s->buf_align;

    bs->bl.opt_mem_alignment = MAX(s->buf_align, getpagesize());

}
