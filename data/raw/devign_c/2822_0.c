static void raw_eject(BlockDriverState *bs, bool eject_flag)

{

    bdrv_eject(bs->file->bs, eject_flag);

}
