void bdrv_disable_copy_on_read(BlockDriverState *bs)

{

    assert(bs->copy_on_read > 0);

    bs->copy_on_read--;

}
