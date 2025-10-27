int bdrv_read_unthrottled(BlockDriverState *bs, int64_t sector_num,

                          uint8_t *buf, int nb_sectors)

{

    bool enabled;

    int ret;



    enabled = bs->io_limits_enabled;

    bs->io_limits_enabled = false;

    ret = bdrv_read(bs, sector_num, buf, nb_sectors);

    bs->io_limits_enabled = enabled;

    return ret;

}
