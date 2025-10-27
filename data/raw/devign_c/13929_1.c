static void bdrv_qed_invalidate_cache(BlockDriverState *bs)
{
    BDRVQEDState *s = bs->opaque;
    bdrv_qed_close(bs);
    memset(s, 0, sizeof(BDRVQEDState));
    bdrv_qed_open(bs, NULL, bs->open_flags, NULL);
}