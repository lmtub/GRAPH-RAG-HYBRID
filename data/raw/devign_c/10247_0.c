void bdrv_set_dirty_bitmap(BdrvDirtyBitmap *bitmap,

                           int64_t cur_sector, int64_t nr_sectors)

{

    assert(bdrv_dirty_bitmap_enabled(bitmap));

    hbitmap_set(bitmap->bitmap, cur_sector, nr_sectors);

}
