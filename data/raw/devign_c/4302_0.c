int bdrv_debug_remove_breakpoint(BlockDriverState *bs, const char *tag)

{

    while (bs && bs->drv && !bs->drv->bdrv_debug_remove_breakpoint) {

        bs = bs->file;

    }



    if (bs && bs->drv && bs->drv->bdrv_debug_remove_breakpoint) {

        return bs->drv->bdrv_debug_remove_breakpoint(bs, tag);

    }



    return -ENOTSUP;

}
