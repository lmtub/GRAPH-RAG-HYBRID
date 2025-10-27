static int raw_get_info(BlockDriverState *bs, BlockDriverInfo *bdi)

{

    return bdrv_get_info(bs->file->bs, bdi);

}
