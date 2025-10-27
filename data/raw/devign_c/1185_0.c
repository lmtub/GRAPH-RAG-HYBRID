static int mig_save_device_dirty(QEMUFile *f, BlkMigDevState *bmds,

                                 int is_async)

{

    BlkMigBlock *blk;

    BlockDriverState *bs = blk_bs(bmds->blk);

    int64_t total_sectors = bmds->total_sectors;

    int64_t sector;

    int nr_sectors;

    int ret = -EIO;



    for (sector = bmds->cur_dirty; sector < bmds->total_sectors;) {

        blk_mig_lock();

        if (bmds_aio_inflight(bmds, sector)) {

            blk_mig_unlock();

            blk_drain(bmds->blk);

        } else {

            blk_mig_unlock();

        }

        if (bdrv_get_dirty(bs, bmds->dirty_bitmap, sector)) {



            if (total_sectors - sector < BDRV_SECTORS_PER_DIRTY_CHUNK) {

                nr_sectors = total_sectors - sector;

            } else {

                nr_sectors = BDRV_SECTORS_PER_DIRTY_CHUNK;

            }

            bdrv_reset_dirty_bitmap(bmds->dirty_bitmap, sector, nr_sectors);



            blk = g_new(BlkMigBlock, 1);

            blk->buf = g_malloc(BLOCK_SIZE);

            blk->bmds = bmds;

            blk->sector = sector;

            blk->nr_sectors = nr_sectors;



            if (is_async) {

                blk->iov.iov_base = blk->buf;

                blk->iov.iov_len = nr_sectors * BDRV_SECTOR_SIZE;

                qemu_iovec_init_external(&blk->qiov, &blk->iov, 1);



                blk->aiocb = blk_aio_preadv(bmds->blk,

                                            sector * BDRV_SECTOR_SIZE,

                                            &blk->qiov, 0, blk_mig_read_cb,

                                            blk);



                blk_mig_lock();

                block_mig_state.submitted++;

                bmds_set_aio_inflight(bmds, sector, nr_sectors, 1);

                blk_mig_unlock();

            } else {

                ret = blk_pread(bmds->blk, sector * BDRV_SECTOR_SIZE, blk->buf,

                                nr_sectors * BDRV_SECTOR_SIZE);

                if (ret < 0) {

                    goto error;

                }

                blk_send(f, blk);



                g_free(blk->buf);

                g_free(blk);

            }



            sector += nr_sectors;

            bmds->cur_dirty = sector;



            break;

        }

        sector += BDRV_SECTORS_PER_DIRTY_CHUNK;

        bmds->cur_dirty = sector;

    }



    return (bmds->cur_dirty >= bmds->total_sectors);



error:

    DPRINTF("Error reading sector %" PRId64 "\n", sector);

    g_free(blk->buf);

    g_free(blk);

    return ret;

}
