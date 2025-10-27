bool bdrv_is_first_non_filter(BlockDriverState *candidate)

{

    BlockDriverState *bs;

    BdrvNextIterator *it = NULL;



    /* walk down the bs forest recursively */

    while ((it = bdrv_next(it, &bs)) != NULL) {

        bool perm;



        /* try to recurse in this top level bs */

        perm = bdrv_recurse_is_first_non_filter(bs, candidate);



        /* candidate is the first non filter */

        if (perm) {

            return true;

        }

    }



    return false;

}
