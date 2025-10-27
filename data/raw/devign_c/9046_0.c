int bdrv_make_zero(BlockDriverState *bs, BdrvRequestFlags flags)

{

    int64_t target_sectors, ret, nb_sectors, sector_num = 0;

    int n;



    target_sectors = bdrv_nb_sectors(bs);

    if (target_sectors < 0) {

        return target_sectors;

    }



    for (;;) {

        nb_sectors = target_sectors - sector_num;

        if (nb_sectors <= 0) {

            return 0;

        }

        if (nb_sectors > INT_MAX / BDRV_SECTOR_SIZE) {

            nb_sectors = INT_MAX / BDRV_SECTOR_SIZE;

        }

        ret = bdrv_get_block_status(bs, sector_num, nb_sectors, &n);

        if (ret < 0) {

            error_report("error getting block status at sector %" PRId64 ": %s",

                         sector_num, strerror(-ret));

            return ret;

        }

        if (ret & BDRV_BLOCK_ZERO) {

            sector_num += n;

            continue;

        }

        ret = bdrv_write_zeroes(bs, sector_num, n, flags);

        if (ret < 0) {

            error_report("error writing zeroes at sector %" PRId64 ": %s",

                         sector_num, strerror(-ret));

            return ret;

        }

        sector_num += n;

    }

}
