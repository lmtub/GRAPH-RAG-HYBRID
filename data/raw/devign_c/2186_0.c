void bdrv_set_in_use(BlockDriverState *bs, int in_use)

{

    assert(bs->in_use != in_use);

    bs->in_use = in_use;

}
