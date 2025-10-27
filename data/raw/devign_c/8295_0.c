static bool raw_is_inserted(BlockDriverState *bs)

{

    return bdrv_is_inserted(bs->file->bs);

}
