ImageInfoSpecific *bdrv_get_specific_info(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (drv && drv->bdrv_get_specific_info) {

        return drv->bdrv_get_specific_info(bs);

    }

    return NULL;

}
