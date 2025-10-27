void bdrv_set_enable_write_cache(BlockDriverState *bs, bool wce)

{

    bs->enable_write_cache = wce;



    /* so a reopen() will preserve wce */

    if (wce) {

        bs->open_flags |= BDRV_O_CACHE_WB;

    } else {

        bs->open_flags &= ~BDRV_O_CACHE_WB;

    }

}
