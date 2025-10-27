static int64_t coroutine_fn vpc_co_get_block_status(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *pnum, BlockDriverState **file)

{

    BDRVVPCState *s = bs->opaque;

    VHDFooter *footer = (VHDFooter*) s->footer_buf;

    int64_t start, offset;

    bool allocated;

    int64_t ret;

    int n;



    if (be32_to_cpu(footer->type) == VHD_FIXED) {

        *pnum = nb_sectors;

        *file = bs->file->bs;

        return BDRV_BLOCK_RAW | BDRV_BLOCK_OFFSET_VALID |

               (sector_num << BDRV_SECTOR_BITS);

    }



    qemu_co_mutex_lock(&s->lock);



    offset = get_image_offset(bs, sector_num << BDRV_SECTOR_BITS, false);

    start = offset;

    allocated = (offset != -1);

    *pnum = 0;

    ret = 0;



    do {

        /* All sectors in a block are contiguous (without using the bitmap) */

        n = ROUND_UP(sector_num + 1, s->block_size / BDRV_SECTOR_SIZE)

          - sector_num;

        n = MIN(n, nb_sectors);



        *pnum += n;

        sector_num += n;

        nb_sectors -= n;

        /* *pnum can't be greater than one block for allocated

         * sectors since there is always a bitmap in between. */

        if (allocated) {

            *file = bs->file->bs;

            ret = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | start;

            break;

        }

        if (nb_sectors == 0) {

            break;

        }

        offset = get_image_offset(bs, sector_num << BDRV_SECTOR_BITS, false);

    } while (offset == -1);



    qemu_co_mutex_unlock(&s->lock);

    return ret;

}
