void laio_io_plug(BlockDriverState *bs, void *aio_ctx)

{

    struct qemu_laio_state *s = aio_ctx;



    s->io_q.plugged++;

}
