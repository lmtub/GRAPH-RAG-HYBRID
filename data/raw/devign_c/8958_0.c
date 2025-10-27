int bdrv_write(BlockDriverState *bs, int64_t sector_num,

               const uint8_t *buf, int nb_sectors)

{

    return bdrv_rw_co(bs, sector_num, (uint8_t *)buf, nb_sectors, true, 0);

}
