void bdrv_set_geometry_hint(BlockDriverState *bs,

                            int cyls, int heads, int secs)

{

    bs->cyls = cyls;

    bs->heads = heads;

    bs->secs = secs;

}
