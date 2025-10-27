static int mig_save_device_dirty(Monitor *mon, QEMUFile *f,

                                 BlkMigDevState *bmds, int is_async)

{

    BlkMigBlock *blk;

    int64_t total_sectors = bmds->total_sectors;

    int64_t sector;

    int nr_sectors;

    int ret = -EIO;



    for (sector = bmds->cur_dirty; sector < bmds->total_sectors;) {

        if (bmds_aio_inflight(bmds, sector)) {

            qemu_aio_flush();

        }

        if (bdrv_get_dirty(bmds->bs, sector)) {



            if (total_sectors - sector < BDRV_SECTORS_PER_DIRTY_CHUNK) {

                nr_sectors = total_sectors - sector;

            } else {

                nr_sectors = BDRV_SECTORS_PER_DIRTY_CHUNK;

            }

            blk = g_malloc(sizeof(BlkMigBlock));

            blk->buf = g_malloc(BLOCK_SIZE);

            blk->bmds = bmds;

            blk->sector = sector;

            blk->nr_sectors = nr_sectors;



            if (is_async) {

                blk->iov.iov_base = blk->buf;

                blk->iov.iov_len = nr_sectors * BDRV_SECTOR_SIZE;

                qemu_iovec_init_external(&blk->qiov, &blk->iov, 1);



                if (block_mig_state.submitted == 0) {

                    block_mig_state.prev_time_offset = qemu_get_clock_ns(rt_clock);

                }



                blk->aiocb = bdrv_aio_readv(bmds->bs, sector, &blk->qiov,

                                            nr_sectors, blk_mig_read_cb, blk);

                if (!blk->aiocb) {

                    goto error;

                }

                block_mig_state.submitted++;

                bmds_set_aio_inflight(bmds, sector, nr_sectors, 1);

            } else {

                ret = bdrv_read(bmds->bs, sector, blk->buf, nr_sectors);

                if (ret < 0) {

                    goto error;

                }

                blk_send(f, blk);



                g_free(blk->buf);

                g_free(blk);

            }



            bdrv_reset_dirty(bmds->bs, sector, nr_sectors);

            break;

        }

        sector += BDRV_SECTORS_PER_DIRTY_CHUNK;

        bmds->cur_dirty = sector;

    }



    return (bmds->cur_dirty >= bmds->total_sectors);



error:

    monitor_printf(mon, "Error reading sector %" PRId64 "\n", sector);

    qemu_file_set_error(f, ret);

    g_free(blk->buf);

    g_free(blk);

    return 0;

}
