static uint64_t coroutine_fn mirror_iteration(MirrorBlockJob *s)

{

    BlockDriverState *source = blk_bs(s->common.blk);

    int64_t sector_num, first_chunk;

    uint64_t delay_ns = 0;

    /* At least the first dirty chunk is mirrored in one iteration. */

    int nb_chunks = 1;

    int64_t end = s->bdev_length / BDRV_SECTOR_SIZE;

    int sectors_per_chunk = s->granularity >> BDRV_SECTOR_BITS;

    bool write_zeroes_ok = bdrv_can_write_zeroes_with_unmap(blk_bs(s->target));



    sector_num = hbitmap_iter_next(&s->hbi);

    if (sector_num < 0) {

        bdrv_dirty_iter_init(s->dirty_bitmap, &s->hbi);

        sector_num = hbitmap_iter_next(&s->hbi);

        trace_mirror_restart_iter(s, bdrv_get_dirty_count(s->dirty_bitmap));

        assert(sector_num >= 0);

    }



    first_chunk = sector_num / sectors_per_chunk;

    while (test_bit(first_chunk, s->in_flight_bitmap)) {

        trace_mirror_yield_in_flight(s, sector_num, s->in_flight);

        mirror_wait_for_io(s);

    }



    block_job_pause_point(&s->common);



    /* Find the number of consective dirty chunks following the first dirty

     * one, and wait for in flight requests in them. */

    while (nb_chunks * sectors_per_chunk < (s->buf_size >> BDRV_SECTOR_BITS)) {

        int64_t hbitmap_next;

        int64_t next_sector = sector_num + nb_chunks * sectors_per_chunk;

        int64_t next_chunk = next_sector / sectors_per_chunk;

        if (next_sector >= end ||

            !bdrv_get_dirty(source, s->dirty_bitmap, next_sector)) {

            break;

        }

        if (test_bit(next_chunk, s->in_flight_bitmap)) {

            break;

        }



        hbitmap_next = hbitmap_iter_next(&s->hbi);

        if (hbitmap_next > next_sector || hbitmap_next < 0) {

            /* The bitmap iterator's cache is stale, refresh it */

            bdrv_set_dirty_iter(&s->hbi, next_sector);

            hbitmap_next = hbitmap_iter_next(&s->hbi);

        }

        assert(hbitmap_next == next_sector);

        nb_chunks++;

    }



    /* Clear dirty bits before querying the block status, because

     * calling bdrv_get_block_status_above could yield - if some blocks are

     * marked dirty in this window, we need to know.

     */

    bdrv_reset_dirty_bitmap(s->dirty_bitmap, sector_num,

                            nb_chunks * sectors_per_chunk);

    bitmap_set(s->in_flight_bitmap, sector_num / sectors_per_chunk, nb_chunks);

    while (nb_chunks > 0 && sector_num < end) {

        int ret;

        int io_sectors, io_sectors_acct;

        BlockDriverState *file;

        enum MirrorMethod {

            MIRROR_METHOD_COPY,

            MIRROR_METHOD_ZERO,

            MIRROR_METHOD_DISCARD

        } mirror_method = MIRROR_METHOD_COPY;



        assert(!(sector_num % sectors_per_chunk));

        ret = bdrv_get_block_status_above(source, NULL, sector_num,

                                          nb_chunks * sectors_per_chunk,

                                          &io_sectors, &file);

        if (ret < 0) {

            io_sectors = nb_chunks * sectors_per_chunk;

        }



        io_sectors -= io_sectors % sectors_per_chunk;

        if (io_sectors < sectors_per_chunk) {

            io_sectors = sectors_per_chunk;

        } else if (ret >= 0 && !(ret & BDRV_BLOCK_DATA)) {

            int64_t target_sector_num;

            int target_nb_sectors;

            bdrv_round_sectors_to_clusters(blk_bs(s->target), sector_num,

                                           io_sectors,  &target_sector_num,

                                           &target_nb_sectors);

            if (target_sector_num == sector_num &&

                target_nb_sectors == io_sectors) {

                mirror_method = ret & BDRV_BLOCK_ZERO ?

                                    MIRROR_METHOD_ZERO :

                                    MIRROR_METHOD_DISCARD;

            }

        }



        while (s->in_flight >= MAX_IN_FLIGHT) {

            trace_mirror_yield_in_flight(s, sector_num, s->in_flight);

            mirror_wait_for_io(s);

        }



        mirror_clip_sectors(s, sector_num, &io_sectors);

        switch (mirror_method) {

        case MIRROR_METHOD_COPY:

            io_sectors = mirror_do_read(s, sector_num, io_sectors);

            io_sectors_acct = io_sectors;

            break;

        case MIRROR_METHOD_ZERO:

        case MIRROR_METHOD_DISCARD:

            mirror_do_zero_or_discard(s, sector_num, io_sectors,

                                      mirror_method == MIRROR_METHOD_DISCARD);

            if (write_zeroes_ok) {

                io_sectors_acct = 0;

            } else {

                io_sectors_acct = io_sectors;

            }

            break;

        default:

            abort();

        }

        assert(io_sectors);

        sector_num += io_sectors;

        nb_chunks -= DIV_ROUND_UP(io_sectors, sectors_per_chunk);

        if (s->common.speed) {

            delay_ns = ratelimit_calculate_delay(&s->limit, io_sectors_acct);

        }

    }

    return delay_ns;

}
