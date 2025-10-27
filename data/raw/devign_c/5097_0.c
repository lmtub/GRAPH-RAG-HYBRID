uint64_t bdrv_dirty_bitmap_serialization_size(const BdrvDirtyBitmap *bitmap,

                                              uint64_t start, uint64_t count)

{

    return hbitmap_serialization_size(bitmap->bitmap, start, count);

}
