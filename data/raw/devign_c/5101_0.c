static int raw_read(BlockDriverState *bs, int64_t sector_num,

                    uint8_t *buf, int nb_sectors)

{

    return bdrv_read(bs->file, sector_num, buf, nb_sectors);

}
