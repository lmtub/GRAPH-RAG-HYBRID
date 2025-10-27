static void set_dirty_bitmap(BlockDriverState *bs, int64_t sector_num,

                             int nb_sectors, int dirty)

{

    int64_t start, end;

    unsigned long val, idx, bit;



    start = sector_num / BDRV_SECTORS_PER_DIRTY_CHUNK;

    end = (sector_num + nb_sectors - 1) / BDRV_SECTORS_PER_DIRTY_CHUNK;



    for (; start <= end; start++) {

        idx = start / (sizeof(unsigned long) * 8);

        bit = start % (sizeof(unsigned long) * 8);

        val = bs->dirty_bitmap[idx];

        if (dirty) {

            val |= 1 << bit;

        } else {

            val &= ~(1 << bit);

        }

        bs->dirty_bitmap[idx] = val;

    }

}
