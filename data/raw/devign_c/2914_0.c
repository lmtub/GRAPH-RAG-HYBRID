QEMUBH *aio_bh_new(AioContext *ctx, QEMUBHFunc *cb, void *opaque)

{

    QEMUBH *bh;

    bh = g_malloc0(sizeof(QEMUBH));

    bh->ctx = ctx;

    bh->cb = cb;

    bh->opaque = opaque;

    qemu_mutex_lock(&ctx->bh_lock);

    bh->next = ctx->first_bh;

    /* Make sure that the members are ready before putting bh into list */

    smp_wmb();

    ctx->first_bh = bh;

    qemu_mutex_unlock(&ctx->bh_lock);

    return bh;

}
