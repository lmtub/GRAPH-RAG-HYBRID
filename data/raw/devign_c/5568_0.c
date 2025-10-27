int bdrv_debug_breakpoint(BlockDriverState *bs, const char *event,

                          const char *tag)

{

    while (bs && bs->drv && !bs->drv->bdrv_debug_breakpoint) {

        bs = bs->file;

    }



    if (bs && bs->drv && bs->drv->bdrv_debug_breakpoint) {

        return bs->drv->bdrv_debug_breakpoint(bs, event, tag);

    }



    return -ENOTSUP;

}
