bool bdrv_debug_is_suspended(BlockDriverState *bs, const char *tag)

{

    while (bs && bs->drv && !bs->drv->bdrv_debug_is_suspended) {

        bs = bs->file;

    }



    if (bs && bs->drv && bs->drv->bdrv_debug_is_suspended) {

        return bs->drv->bdrv_debug_is_suspended(bs, tag);

    }



    return false;

}
