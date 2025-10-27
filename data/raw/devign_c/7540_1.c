static int cow_read(BlockDriverState *bs, int64_t sector_num,

                    uint8_t *buf, int nb_sectors)

{

    BDRVCowState *s = bs->opaque;

    int ret, n;



    while (nb_sectors > 0) {

        if (cow_is_allocated(bs, sector_num, nb_sectors, &n)) {

            ret = bdrv_pread(bs->file,

                        s->cow_sectors_offset + sector_num * 512,

                        buf, n * 512);

            if (ret != n * 512)

                return -1;

        } else {

            if (bs->backing_hd) {

                /* read from the base image */

                ret = bdrv_read(bs->backing_hd, sector_num, buf, n);

                if (ret < 0)

                    return -1;

            } else {

            memset(buf, 0, n * 512);

        }

        }

        nb_sectors -= n;

        sector_num += n;

        buf += n * 512;

    }

    return 0;

}
