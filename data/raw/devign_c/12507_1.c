static int qcow2_check(BlockDriverState *bs, BdrvCheckResult *result,

                       BdrvCheckMode fix)

{

    int ret = qcow2_check_refcounts(bs, result, fix);

    if (ret < 0) {

        return ret;

    }



    if (fix && result->check_errors == 0 && result->corruptions == 0) {

        return qcow2_mark_clean(bs);

    }

    return ret;

}
