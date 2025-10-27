static void raw_aio_writev_scrubbed(void *opaque, int ret)

{

    RawScrubberBounce *b = opaque;



    if (ret < 0) {

        b->cb(b->opaque, ret);

    } else {

        b->cb(b->opaque, ret + 512);

    }



    qemu_iovec_destroy(&b->qiov);

    qemu_free(b);

}
