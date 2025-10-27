static int raw_write(BlockDriverState *bs, int64_t sector_num,

                     const uint8_t *buf, int nb_sectors)

{

    return bdrv_write(bs->file, sector_num, buf, nb_sectors);

}
