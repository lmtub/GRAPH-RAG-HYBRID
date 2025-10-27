static int bdrv_rd_badreq_sectors(BlockDriverState *bs,

                                  int64_t sector_num, int nb_sectors)

{

    return

        nb_sectors < 0 ||

        sector_num < 0 ||

        nb_sectors > bs->total_sectors ||

        sector_num > bs->total_sectors - nb_sectors;

}
