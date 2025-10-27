void bdrv_get_geometry_hint(BlockDriverState *bs,

                            int *pcyls, int *pheads, int *psecs)

{

    *pcyls = bs->cyls;

    *pheads = bs->heads;

    *psecs = bs->secs;

}
