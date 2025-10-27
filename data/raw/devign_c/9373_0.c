int bdrv_get_dirty(BlockDriverState *bs, BdrvDirtyBitmap *bitmap,

                   int64_t sector)

{

    if (bitmap) {

        return hbitmap_get(bitmap->bitmap, sector);

    } else {

        return 0;

    }

}
