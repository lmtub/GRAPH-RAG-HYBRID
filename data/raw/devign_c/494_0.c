int64_t bdrv_dirty_iter_next(BdrvDirtyBitmapIter *iter)

{

    return hbitmap_iter_next(&iter->hbi);

}
