int bdrv_key_required(BlockDriverState *bs)

{

    BlockDriverState *backing_hd = bs->backing_hd;



    if (backing_hd && backing_hd->encrypted && !backing_hd->valid_key)

        return 1;

    return (bs->encrypted && !bs->valid_key);

}
