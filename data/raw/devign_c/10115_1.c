static int vhdx_create_bat(BlockDriverState *bs, BDRVVHDXState *s,

                           uint64_t image_size, VHDXImageType type,

                           bool use_zero_blocks, uint64_t file_offset,

                           uint32_t length)

{

    int ret = 0;

    uint64_t data_file_offset;

    uint64_t total_sectors = 0;

    uint64_t sector_num = 0;

    uint64_t unused;

    int block_state;

    VHDXSectorInfo sinfo;



    assert(s->bat == NULL);



    /* this gives a data start after BAT/bitmap entries, and well

     * past any metadata entries (with a 4 MB buffer for future

     * expansion */

    data_file_offset = file_offset + length + 5 * MiB;

    total_sectors = image_size >> s->logical_sector_size_bits;



    if (type == VHDX_TYPE_DYNAMIC) {

        /* All zeroes, so we can just extend the file - the end of the BAT

         * is the furthest thing we have written yet */

        ret = bdrv_truncate(bs, data_file_offset);

        if (ret < 0) {

            goto exit;

        }

    } else if (type == VHDX_TYPE_FIXED) {

        ret = bdrv_truncate(bs, data_file_offset + image_size);

        if (ret < 0) {

            goto exit;

        }

    } else {

        ret = -ENOTSUP;

        goto exit;

    }



    if (type == VHDX_TYPE_FIXED ||

                use_zero_blocks ||

                bdrv_has_zero_init(bs) == 0) {

        /* for a fixed file, the default BAT entry is not zero */

        s->bat = g_malloc0(length);

        block_state = type == VHDX_TYPE_FIXED ? PAYLOAD_BLOCK_FULLY_PRESENT :

                                                PAYLOAD_BLOCK_NOT_PRESENT;

        block_state = use_zero_blocks ? PAYLOAD_BLOCK_ZERO : block_state;

        /* fill the BAT by emulating sector writes of sectors_per_block size */

        while (sector_num < total_sectors) {

            vhdx_block_translate(s, sector_num, s->sectors_per_block, &sinfo);

            sinfo.file_offset = data_file_offset +

                                (sector_num << s->logical_sector_size_bits);

            sinfo.file_offset = ROUND_UP(sinfo.file_offset, MiB);

            vhdx_update_bat_table_entry(bs, s, &sinfo, &unused, &unused,

                                        block_state);

            cpu_to_le64s(&s->bat[sinfo.bat_idx]);

            sector_num += s->sectors_per_block;

        }

        ret = bdrv_pwrite(bs, file_offset, s->bat, length);

        if (ret < 0) {

            goto exit;

        }

    }







exit:

    g_free(s->bat);

    return ret;

}
