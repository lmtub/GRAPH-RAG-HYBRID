int bdrv_get_backing_file_depth(BlockDriverState *bs)

{

    if (!bs->drv) {

        return 0;

    }



    if (!bs->backing_hd) {

        return 0;

    }



    return 1 + bdrv_get_backing_file_depth(bs->backing_hd);

}
