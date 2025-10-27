static int mirror_cow_align(MirrorBlockJob *s,

                            int64_t *sector_num,

                            int *nb_sectors)

{

    bool need_cow;

    int ret = 0;

    int chunk_sectors = s->granularity >> BDRV_SECTOR_BITS;

    int64_t align_sector_num = *sector_num;

    int align_nb_sectors = *nb_sectors;

    int max_sectors = chunk_sectors * s->max_iov;



    need_cow = !test_bit(*sector_num / chunk_sectors, s->cow_bitmap);

    need_cow |= !test_bit((*sector_num + *nb_sectors - 1) / chunk_sectors,

                          s->cow_bitmap);

    if (need_cow) {

        bdrv_round_sectors_to_clusters(blk_bs(s->target), *sector_num,

                                       *nb_sectors, &align_sector_num,

                                       &align_nb_sectors);

    }



    if (align_nb_sectors > max_sectors) {

        align_nb_sectors = max_sectors;

        if (need_cow) {

            align_nb_sectors = QEMU_ALIGN_DOWN(align_nb_sectors,

                                               s->target_cluster_sectors);

        }

    }

    /* Clipping may result in align_nb_sectors unaligned to chunk boundary, but

     * that doesn't matter because it's already the end of source image. */

    mirror_clip_sectors(s, align_sector_num, &align_nb_sectors);



    ret = align_sector_num + align_nb_sectors - (*sector_num + *nb_sectors);

    *sector_num = align_sector_num;

    *nb_sectors = align_nb_sectors;

    assert(ret >= 0);

    return ret;

}
