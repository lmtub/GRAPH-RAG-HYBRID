void bdrv_dirty_bitmap_deserialize_part(BdrvDirtyBitmap *bitmap,

                                        uint8_t *buf, uint64_t start,

                                        uint64_t count, bool finish)

{

    hbitmap_deserialize_part(bitmap->bitmap, buf, start, count, finish);

}
