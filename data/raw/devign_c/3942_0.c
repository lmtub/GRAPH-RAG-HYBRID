static void mirror_do_zero_or_discard(MirrorBlockJob *s,

                                      int64_t sector_num,

                                      int nb_sectors,

                                      bool is_discard)

{

    MirrorOp *op;



    /* Allocate a MirrorOp that is used as an AIO callback. The qiov is zeroed

     * so the freeing in mirror_iteration_done is nop. */

    op = g_new0(MirrorOp, 1);

    op->s = s;

    op->sector_num = sector_num;

    op->nb_sectors = nb_sectors;



    s->in_flight++;

    s->sectors_in_flight += nb_sectors;

    if (is_discard) {

        blk_aio_pdiscard(s->target, sector_num << BDRV_SECTOR_BITS,

                         op->nb_sectors << BDRV_SECTOR_BITS,

                         mirror_write_complete, op);

    } else {

        blk_aio_pwrite_zeroes(s->target, sector_num * BDRV_SECTOR_SIZE,

                              op->nb_sectors * BDRV_SECTOR_SIZE,

                              s->unmap ? BDRV_REQ_MAY_UNMAP : 0,

                              mirror_write_complete, op);

    }

}
