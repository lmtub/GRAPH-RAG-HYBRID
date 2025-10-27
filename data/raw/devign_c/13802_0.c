bochs_co_preadv(BlockDriverState *bs, uint64_t offset, uint64_t bytes,

                QEMUIOVector *qiov, int flags)

{

    BDRVBochsState *s = bs->opaque;

    uint64_t sector_num = offset >> BDRV_SECTOR_BITS;

    int nb_sectors = bytes >> BDRV_SECTOR_BITS;

    uint64_t bytes_done = 0;

    QEMUIOVector local_qiov;

    int ret;



    assert((offset & (BDRV_SECTOR_SIZE - 1)) == 0);

    assert((bytes & (BDRV_SECTOR_SIZE - 1)) == 0);



    qemu_iovec_init(&local_qiov, qiov->niov);

    qemu_co_mutex_lock(&s->lock);



    while (nb_sectors > 0) {

        int64_t block_offset = seek_to_sector(bs, sector_num);

        if (block_offset < 0) {

            ret = block_offset;

            goto fail;

        }



        qemu_iovec_reset(&local_qiov);

        qemu_iovec_concat(&local_qiov, qiov, bytes_done, 512);



        if (block_offset > 0) {

            ret = bdrv_co_preadv(bs->file->bs, block_offset, 512,

                                 &local_qiov, 0);

            if (ret < 0) {

                goto fail;

            }

        } else {

            qemu_iovec_memset(&local_qiov, 0, 0, 512);

        }

        nb_sectors--;

        sector_num++;

        bytes_done += 512;

    }



    ret = 0;

fail:

    qemu_co_mutex_unlock(&s->lock);

    qemu_iovec_destroy(&local_qiov);



    return ret;

}
