static int qcow2_check(BlockDriverState *bs, BdrvCheckResult *result,

                       BdrvCheckMode fix)

{

    return qcow2_check_refcounts(bs, result, fix);

}
