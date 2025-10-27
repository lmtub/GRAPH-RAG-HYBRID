static int coroutine_fn is_allocated_base(BlockDriverState *top,

                                          BlockDriverState *base,

                                          int64_t sector_num,

                                          int nb_sectors, int *pnum)

{

    BlockDriverState *intermediate;

    int ret, n;



    ret = bdrv_co_is_allocated(top, sector_num, nb_sectors, &n);

    if (ret) {

        *pnum = n;

        return ret;

    }



    /*

     * Is the unallocated chunk [sector_num, n] also

     * unallocated between base and top?

     */

    intermediate = top->backing_hd;



    while (intermediate != base) {

        int pnum_inter;



        ret = bdrv_co_is_allocated(intermediate, sector_num, nb_sectors,

                                   &pnum_inter);

        if (ret < 0) {

            return ret;

        } else if (ret) {

            *pnum = pnum_inter;

            return 0;

        }



        /*

         * [sector_num, nb_sectors] is unallocated on top but intermediate

         * might have

         *

         * [sector_num+x, nr_sectors] allocated.

         */

        if (n > pnum_inter) {

            n = pnum_inter;

        }



        intermediate = intermediate->backing_hd;

    }



    *pnum = n;

    return 1;

}
