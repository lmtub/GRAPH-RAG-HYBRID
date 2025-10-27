int bdrv_make_zero(BdrvChild *child, BdrvRequestFlags flags)

{

    int64_t target_size, ret, bytes, offset = 0;

    BlockDriverState *bs = child->bs;

    int n; /* sectors */



    target_size = bdrv_getlength(bs);

    if (target_size < 0) {

        return target_size;

    }



    for (;;) {

        bytes = MIN(target_size - offset, BDRV_REQUEST_MAX_BYTES);

        if (bytes <= 0) {

            return 0;

        }

        ret = bdrv_get_block_status(bs, offset >> BDRV_SECTOR_BITS,

                                    bytes >> BDRV_SECTOR_BITS, &n, NULL);

        if (ret < 0) {

            error_report("error getting block status at offset %" PRId64 ": %s",

                         offset, strerror(-ret));

            return ret;

        }

        if (ret & BDRV_BLOCK_ZERO) {

            offset += n * BDRV_SECTOR_BITS;

            continue;

        }

        ret = bdrv_pwrite_zeroes(child, offset, n * BDRV_SECTOR_SIZE, flags);

        if (ret < 0) {

            error_report("error writing zeroes at offset %" PRId64 ": %s",

                         offset, strerror(-ret));

            return ret;

        }

        offset += n * BDRV_SECTOR_SIZE;

    }

}
