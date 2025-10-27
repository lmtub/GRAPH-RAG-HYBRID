static void bdrv_qed_refresh_limits(BlockDriverState *bs, Error **errp)

{

    BDRVQEDState *s = bs->opaque;



    bs->bl.write_zeroes_alignment = s->header.cluster_size >> BDRV_SECTOR_BITS;

}
