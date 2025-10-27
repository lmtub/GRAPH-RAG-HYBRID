static int cow_write(BlockDriverState *bs, int64_t sector_num,

                     const uint8_t *buf, int nb_sectors)

{

    BDRVCowState *s = bs->opaque;

    int ret;



    ret = bdrv_pwrite(bs->file, s->cow_sectors_offset + sector_num * 512,

                      buf, nb_sectors * 512);

    if (ret < 0) {

        return ret;

    }



    return cow_update_bitmap(bs, sector_num, nb_sectors);

}
