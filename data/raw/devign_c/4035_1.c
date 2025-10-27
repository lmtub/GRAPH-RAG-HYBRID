static int64_t alloc_block(BlockDriverState* bs, int64_t offset)

{

    BDRVVPCState *s = bs->opaque;

    int64_t bat_offset;

    uint32_t index, bat_value;

    int ret;

    uint8_t bitmap[s->bitmap_size];



    /* Check if sector_num is valid */

    if ((offset < 0) || (offset > bs->total_sectors * BDRV_SECTOR_SIZE)) {

        return -EINVAL;

    }



    /* Write entry into in-memory BAT */

    index = offset / s->block_size;

    assert(s->pagetable[index] == 0xFFFFFFFF);

    s->pagetable[index] = s->free_data_block_offset / 512;



    /* Initialize the block's bitmap */

    memset(bitmap, 0xff, s->bitmap_size);

    ret = bdrv_pwrite_sync(bs->file, s->free_data_block_offset, bitmap,

        s->bitmap_size);

    if (ret < 0) {

        return ret;

    }



    /* Write new footer (the old one will be overwritten) */

    s->free_data_block_offset += s->block_size + s->bitmap_size;

    ret = rewrite_footer(bs);

    if (ret < 0)

        goto fail;



    /* Write BAT entry to disk */

    bat_offset = s->bat_offset + (4 * index);

    bat_value = cpu_to_be32(s->pagetable[index]);

    ret = bdrv_pwrite_sync(bs->file, bat_offset, &bat_value, 4);

    if (ret < 0)

        goto fail;



    return get_image_offset(bs, offset, false);



fail:

    s->free_data_block_offset -= (s->block_size + s->bitmap_size);

    return ret;

}
