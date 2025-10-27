static int raw_probe_blocksizes(BlockDriverState *bs, BlockSizes *bsz)

{

    BDRVRawState *s = bs->opaque;

    int ret;



    ret = bdrv_probe_blocksizes(bs->file->bs, bsz);

    if (ret < 0) {

        return ret;

    }



    if (!QEMU_IS_ALIGNED(s->offset, MAX(bsz->log, bsz->phys))) {

        return -ENOTSUP;

    }



    return 0;

}
