uint64_t bdrv_dirty_bitmap_serialization_align(const BdrvDirtyBitmap *bitmap)

{

    return hbitmap_serialization_align(bitmap->bitmap);

}
