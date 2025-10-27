static int get_block_status(BlockDriverState *bs, int64_t sector_num,

                            int nb_sectors, MapEntry *e)

{

    int64_t ret;

    int depth;



    /* As an optimization, we could cache the current range of unallocated

     * clusters in each file of the chain, and avoid querying the same

     * range repeatedly.

     */



    depth = 0;

    for (;;) {

        ret = bdrv_get_block_status(bs, sector_num, nb_sectors, &nb_sectors);

        if (ret < 0) {

            return ret;

        }

        assert(nb_sectors);

        if (ret & (BDRV_BLOCK_ZERO|BDRV_BLOCK_DATA)) {

            break;

        }

        bs = backing_bs(bs);

        if (bs == NULL) {

            ret = 0;

            break;

        }



        depth++;

    }



    e->start = sector_num * BDRV_SECTOR_SIZE;

    e->length = nb_sectors * BDRV_SECTOR_SIZE;

    e->flags = ret & ~BDRV_BLOCK_OFFSET_MASK;

    e->offset = ret & BDRV_BLOCK_OFFSET_MASK;

    e->depth = depth;

    e->bs = bs;

    return 0;

}
