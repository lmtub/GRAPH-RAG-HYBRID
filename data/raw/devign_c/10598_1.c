BlockDriverState *bdrv_new(void)
{
    BlockDriverState *bs;
    int i;
    bs = g_new0(BlockDriverState, 1);
    QLIST_INIT(&bs->dirty_bitmaps);
    for (i = 0; i < BLOCK_OP_TYPE_MAX; i++) {
        QLIST_INIT(&bs->op_blockers[i]);
    }
    notifier_with_return_list_init(&bs->before_write_notifiers);
    bs->refcnt = 1;
    bs->aio_context = qemu_get_aio_context();
    QTAILQ_INSERT_TAIL(&all_bdrv_states, bs, bs_list);
    return bs;
}