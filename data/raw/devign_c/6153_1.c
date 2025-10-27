int bdrv_has_zero_init(BlockDriverState *bs)

{

    assert(bs->drv);



    if (bs->drv->bdrv_has_zero_init) {

        return bs->drv->bdrv_has_zero_init(bs);

    }



    return 1;

}
