void bdrv_set_dirty(BlockDriverState *bs, int64_t cur_sector,

                    int nr_sectors)

{

    BdrvDirtyBitmap *bitmap;

    QLIST_FOREACH(bitmap, &bs->dirty_bitmaps, list) {

        hbitmap_set(bitmap->bitmap, cur_sector, nr_sectors);

    }

}
