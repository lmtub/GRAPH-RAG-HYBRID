static int vpc_open(BlockDriverState *bs, QDict *options, int flags,

                    Error **errp)

{

    BDRVVPCState *s = bs->opaque;

    int i;

    VHDFooter *footer;

    VHDDynDiskHeader *dyndisk_header;

    uint8_t buf[HEADER_SIZE];

    uint32_t checksum;

    int disk_type = VHD_DYNAMIC;

    int ret;



    ret = bdrv_pread(bs->file, 0, s->footer_buf, HEADER_SIZE);

    if (ret < 0) {

        goto fail;

    }



    footer = (VHDFooter *) s->footer_buf;

    if (strncmp(footer->creator, "conectix", 8)) {

        int64_t offset = bdrv_getlength(bs->file);

        if (offset < 0) {

            ret = offset;

            goto fail;

        } else if (offset < HEADER_SIZE) {

            ret = -EINVAL;

            goto fail;

        }



        /* If a fixed disk, the footer is found only at the end of the file */

        ret = bdrv_pread(bs->file, offset-HEADER_SIZE, s->footer_buf,

                         HEADER_SIZE);

        if (ret < 0) {

            goto fail;

        }

        if (strncmp(footer->creator, "conectix", 8)) {

            error_setg(errp, "invalid VPC image");

            ret = -EINVAL;

            goto fail;

        }

        disk_type = VHD_FIXED;

    }



    checksum = be32_to_cpu(footer->checksum);

    footer->checksum = 0;

    if (vpc_checksum(s->footer_buf, HEADER_SIZE) != checksum)

        fprintf(stderr, "block-vpc: The header checksum of '%s' is "

            "incorrect.\n", bs->filename);



    /* Write 'checksum' back to footer, or else will leave it with zero. */

    footer->checksum = be32_to_cpu(checksum);



    // The visible size of a image in Virtual PC depends on the geometry

    // rather than on the size stored in the footer (the size in the footer

    // is too large usually)

    bs->total_sectors = (int64_t)

        be16_to_cpu(footer->cyls) * footer->heads * footer->secs_per_cyl;



    /* images created with disk2vhd report a far higher virtual size

     * than expected with the cyls * heads * sectors_per_cyl formula.

     * use the footer->size instead if the image was created with

     * disk2vhd.

     */

    if (!strncmp(footer->creator_app, "d2v", 4)) {

        bs->total_sectors = be64_to_cpu(footer->size) / BDRV_SECTOR_SIZE;

    }



    /* Allow a maximum disk size of approximately 2 TB */

    if (bs->total_sectors >= 65535LL * 255 * 255) {

        ret = -EFBIG;

        goto fail;

    }



    if (disk_type == VHD_DYNAMIC) {

        ret = bdrv_pread(bs->file, be64_to_cpu(footer->data_offset), buf,

                         HEADER_SIZE);

        if (ret < 0) {

            goto fail;

        }



        dyndisk_header = (VHDDynDiskHeader *) buf;



        if (strncmp(dyndisk_header->magic, "cxsparse", 8)) {

            ret = -EINVAL;

            goto fail;

        }



        s->block_size = be32_to_cpu(dyndisk_header->block_size);

        s->bitmap_size = ((s->block_size / (8 * 512)) + 511) & ~511;



        s->max_table_entries = be32_to_cpu(dyndisk_header->max_table_entries);

        s->pagetable = g_malloc(s->max_table_entries * 4);



        s->bat_offset = be64_to_cpu(dyndisk_header->table_offset);



        ret = bdrv_pread(bs->file, s->bat_offset, s->pagetable,

                         s->max_table_entries * 4);

        if (ret < 0) {

            goto fail;

        }



        s->free_data_block_offset =

            (s->bat_offset + (s->max_table_entries * 4) + 511) & ~511;



        for (i = 0; i < s->max_table_entries; i++) {

            be32_to_cpus(&s->pagetable[i]);

            if (s->pagetable[i] != 0xFFFFFFFF) {

                int64_t next = (512 * (int64_t) s->pagetable[i]) +

                    s->bitmap_size + s->block_size;



                if (next > s->free_data_block_offset) {

                    s->free_data_block_offset = next;

                }

            }

        }



        if (s->free_data_block_offset > bdrv_getlength(bs->file)) {

            error_setg(errp, "block-vpc: free_data_block_offset points after "

                             "the end of file. The image has been truncated.");

            ret = -EINVAL;

            goto fail;

        }



        s->last_bitmap_offset = (int64_t) -1;



#ifdef CACHE

        s->pageentry_u8 = g_malloc(512);

        s->pageentry_u32 = s->pageentry_u8;

        s->pageentry_u16 = s->pageentry_u8;

        s->last_pagetable = -1;

#endif

    }



    qemu_co_mutex_init(&s->lock);



    /* Disable migration when VHD images are used */

    error_set(&s->migration_blocker,

              QERR_BLOCK_FORMAT_FEATURE_NOT_SUPPORTED,

              "vpc", bs->device_name, "live migration");

    migrate_add_blocker(s->migration_blocker);



    return 0;



fail:

    g_free(s->pagetable);

#ifdef CACHE

    g_free(s->pageentry_u8);

#endif

    return ret;

}
