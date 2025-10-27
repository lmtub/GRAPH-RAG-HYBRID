void bdrv_round_to_clusters(BlockDriverState *bs,

                            int64_t sector_num, int nb_sectors,

                            int64_t *cluster_sector_num,

                            int *cluster_nb_sectors)

{

    BlockDriverInfo bdi;



    if (bdrv_get_info(bs, &bdi) < 0 || bdi.cluster_size == 0) {

        *cluster_sector_num = sector_num;

        *cluster_nb_sectors = nb_sectors;

    } else {

        int64_t c = bdi.cluster_size / BDRV_SECTOR_SIZE;

        *cluster_sector_num = QEMU_ALIGN_DOWN(sector_num, c);

        *cluster_nb_sectors = QEMU_ALIGN_UP(sector_num - *cluster_sector_num +

                                            nb_sectors, c);

    }

}
