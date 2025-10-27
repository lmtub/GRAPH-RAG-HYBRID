static int64_t coroutine_fn raw_co_get_block_status(BlockDriverState *bs,

                                            int64_t sector_num,

                                            int nb_sectors, int *pnum)

{

    *pnum = nb_sectors;

    return BDRV_BLOCK_RAW | BDRV_BLOCK_OFFSET_VALID | BDRV_BLOCK_DATA |

           (sector_num << BDRV_SECTOR_BITS);

}
