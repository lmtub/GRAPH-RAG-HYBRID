static int64_t coroutine_fn bdrv_qed_co_get_block_status(BlockDriverState *bs,

                                                 int64_t sector_num,

                                                 int nb_sectors, int *pnum)

{

    BDRVQEDState *s = bs->opaque;

    size_t len = (size_t)nb_sectors * BDRV_SECTOR_SIZE;

    QEDIsAllocatedCB cb = {

        .bs = bs,

        .pos = (uint64_t)sector_num * BDRV_SECTOR_SIZE,

        .status = BDRV_BLOCK_OFFSET_MASK,

        .pnum = pnum,

    };

    QEDRequest request = { .l2_table = NULL };



    qed_find_cluster(s, &request, cb.pos, len, qed_is_allocated_cb, &cb);



    /* Now sleep if the callback wasn't invoked immediately */

    while (cb.status == BDRV_BLOCK_OFFSET_MASK) {

        cb.co = qemu_coroutine_self();

        qemu_coroutine_yield();

    }



    qed_unref_l2_cache_entry(request.l2_table);



    return cb.status;

}
