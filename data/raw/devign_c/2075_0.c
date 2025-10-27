sd_co_get_block_status(BlockDriverState *bs, int64_t sector_num, int nb_sectors,

                       int *pnum)

{

    BDRVSheepdogState *s = bs->opaque;

    SheepdogInode *inode = &s->inode;

    uint32_t object_size = (UINT32_C(1) << inode->block_size_shift);

    uint64_t offset = sector_num * BDRV_SECTOR_SIZE;

    unsigned long start = offset / object_size,

                  end = DIV_ROUND_UP((sector_num + nb_sectors) *

                                     BDRV_SECTOR_SIZE, object_size);

    unsigned long idx;

    int64_t ret = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | offset;



    for (idx = start; idx < end; idx++) {

        if (inode->data_vdi_id[idx] == 0) {

            break;

        }

    }

    if (idx == start) {

        /* Get the longest length of unallocated sectors */

        ret = 0;

        for (idx = start + 1; idx < end; idx++) {

            if (inode->data_vdi_id[idx] != 0) {

                break;

            }

        }

    }



    *pnum = (idx - start) * object_size / BDRV_SECTOR_SIZE;

    if (*pnum > nb_sectors) {

        *pnum = nb_sectors;

    }

    return ret;

}
