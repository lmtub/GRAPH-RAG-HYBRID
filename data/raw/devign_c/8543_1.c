static int coroutine_fn cow_co_is_allocated(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *num_same)

{

    int64_t bitnum = sector_num + sizeof(struct cow_header_v2) * 8;

    uint64_t offset = (bitnum / 8) & -BDRV_SECTOR_SIZE;

    uint8_t bitmap[BDRV_SECTOR_SIZE];

    int ret;

    int changed;



    ret = bdrv_pread(bs->file, offset, &bitmap, sizeof(bitmap));

    if (ret < 0) {

        return ret;

    }



    bitnum &= BITS_PER_BITMAP_SECTOR - 1;

    changed = cow_test_bit(bitnum, bitmap);

    *num_same = cow_find_streak(bitmap, changed, bitnum, nb_sectors);

    return changed;

}
