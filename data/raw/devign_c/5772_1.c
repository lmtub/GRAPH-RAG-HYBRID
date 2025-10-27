void bdrv_reset_dirty(BlockDriverState *bs, int64_t cur_sector, int nr_sectors)

{

    BdrvDirtyBitmap *bitmap;

    QLIST_FOREACH(bitmap, &bs->dirty_bitmaps, list) {

        hbitmap_reset(bitmap->bitmap, cur_sector, nr_sectors);

    }

}
