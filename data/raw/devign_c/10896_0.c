static int bdrv_qed_check(BlockDriverState *bs, BdrvCheckResult *result)

{

    BDRVQEDState *s = bs->opaque;



    return qed_check(s, result, false);

}
