static int bmds_aio_inflight(BlkMigDevState *bmds, int64_t sector)

{

    int64_t chunk = sector / (int64_t)BDRV_SECTORS_PER_DIRTY_CHUNK;



    if (sector < bdrv_nb_sectors(bmds->bs)) {

        return !!(bmds->aio_bitmap[chunk / (sizeof(unsigned long) * 8)] &

            (1UL << (chunk % (sizeof(unsigned long) * 8))));

    } else {

        return 0;

    }

}
