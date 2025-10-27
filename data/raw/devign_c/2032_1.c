static int qcow_check(BlockDriverState *bs)

{

    return qcow2_check_refcounts(bs);

}
