static void bdrv_dirty_bitmap_truncate(BlockDriverState *bs)

{

    BdrvDirtyBitmap *bitmap;

    uint64_t size = bdrv_nb_sectors(bs);



    QLIST_FOREACH(bitmap, &bs->dirty_bitmaps, list) {

        if (bdrv_dirty_bitmap_frozen(bitmap)) {

            continue;

        }

        hbitmap_truncate(bitmap->bitmap, size);

        bitmap->size = size;

    }

}
