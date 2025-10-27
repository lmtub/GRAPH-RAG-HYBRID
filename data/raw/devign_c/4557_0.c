static int bdrv_qed_is_allocated(BlockDriverState *bs, int64_t sector_num,

                                  int nb_sectors, int *pnum)

{

    BDRVQEDState *s = bs->opaque;

    uint64_t pos = (uint64_t)sector_num * BDRV_SECTOR_SIZE;

    size_t len = (size_t)nb_sectors * BDRV_SECTOR_SIZE;

    QEDIsAllocatedCB cb = {

        .is_allocated = -1,

        .pnum = pnum,

    };

    QEDRequest request = { .l2_table = NULL };



    async_context_push();



    qed_find_cluster(s, &request, pos, len, qed_is_allocated_cb, &cb);



    while (cb.is_allocated == -1) {

        qemu_aio_wait();

    }



    async_context_pop();



    qed_unref_l2_cache_entry(request.l2_table);



    return cb.is_allocated;

}
