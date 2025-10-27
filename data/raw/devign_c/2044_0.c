void bdrv_release_named_dirty_bitmaps(BlockDriverState *bs)

{

    bdrv_do_release_matching_dirty_bitmap(bs, NULL, true);

}
