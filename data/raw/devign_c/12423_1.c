static int rewrite_footer(BlockDriverState* bs)

{

    int ret;

    BDRVVPCState *s = bs->opaque;

    int64_t offset = s->free_data_block_offset;



    ret = bdrv_pwrite(bs->file, offset, s->footer_buf, HEADER_SIZE);

    if (ret < 0)

        return ret;



    return 0;

}
