static int coroutine_fn mirror_dirty_init(MirrorBlockJob *s)

{

    int64_t sector_num, end;

    BlockDriverState *base = s->base;

    BlockDriverState *bs = blk_bs(s->common.blk);

    BlockDriverState *target_bs = blk_bs(s->target);

    int ret, n;



    end = s->bdev_length / BDRV_SECTOR_SIZE;



    if (base == NULL && !bdrv_has_zero_init(target_bs)) {

        if (!bdrv_can_write_zeroes_with_unmap(target_bs)) {

            bdrv_set_dirty_bitmap(s->dirty_bitmap, 0, end);

            return 0;

        }



        for (sector_num = 0; sector_num < end; ) {

            int nb_sectors = MIN(end - sector_num,

                QEMU_ALIGN_DOWN(INT_MAX, s->granularity) >> BDRV_SECTOR_BITS);



            mirror_throttle(s);



            if (block_job_is_cancelled(&s->common)) {

                return 0;

            }



            if (s->in_flight >= MAX_IN_FLIGHT) {

                trace_mirror_yield(s, s->in_flight, s->buf_free_count, -1);

                mirror_wait_for_io(s);

                continue;

            }



            mirror_do_zero_or_discard(s, sector_num, nb_sectors, false);

            sector_num += nb_sectors;

        }



        mirror_drain(s);

    }



    /* First part, loop on the sectors and initialize the dirty bitmap.  */

    for (sector_num = 0; sector_num < end; ) {

        /* Just to make sure we are not exceeding int limit. */

        int nb_sectors = MIN(INT_MAX >> BDRV_SECTOR_BITS,

                             end - sector_num);



        mirror_throttle(s);



        if (block_job_is_cancelled(&s->common)) {

            return 0;

        }



        ret = bdrv_is_allocated_above(bs, base, sector_num, nb_sectors, &n);

        if (ret < 0) {

            return ret;

        }



        assert(n > 0);

        if (ret == 1) {

            bdrv_set_dirty_bitmap(s->dirty_bitmap, sector_num, n);

        }

        sector_num += n;

    }

    return 0;

}
