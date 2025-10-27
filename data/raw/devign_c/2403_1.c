static int vhdx_log_flush(BlockDriverState *bs, BDRVVHDXState *s,

                          VHDXLogSequence *logs)

{

    int ret = 0;

    int i;

    uint32_t cnt, sectors_read;

    uint64_t new_file_size;

    void *data = NULL;

    VHDXLogDescEntries *desc_entries = NULL;

    VHDXLogEntryHeader hdr_tmp = { 0 };



    cnt = logs->count;



    data = qemu_blockalign(bs, VHDX_LOG_SECTOR_SIZE);



    ret = vhdx_user_visible_write(bs, s);

    if (ret < 0) {

        goto exit;

    }



    /* each iteration represents one log sequence, which may span multiple

     * sectors */

    while (cnt--) {

        ret = vhdx_log_peek_hdr(bs, &logs->log, &hdr_tmp);

        if (ret < 0) {

            goto exit;

        }

        /* if the log shows a FlushedFileOffset larger than our current file

         * size, then that means the file has been truncated / corrupted, and

         * we must refused to open it / use it */

        if (hdr_tmp.flushed_file_offset > bdrv_getlength(bs->file->bs)) {

            ret = -EINVAL;

            goto exit;

        }



        ret = vhdx_log_read_desc(bs, s, &logs->log, &desc_entries, true);

        if (ret < 0) {

            goto exit;

        }



        for (i = 0; i < desc_entries->hdr.descriptor_count; i++) {

            if (desc_entries->desc[i].signature == VHDX_LOG_DESC_SIGNATURE) {

                /* data sector, so read a sector to flush */

                ret = vhdx_log_read_sectors(bs, &logs->log, &sectors_read,

                                            data, 1, false);

                if (ret < 0) {

                    goto exit;

                }

                if (sectors_read != 1) {

                    ret = -EINVAL;

                    goto exit;

                }

                vhdx_log_data_le_import(data);

            }



            ret = vhdx_log_flush_desc(bs, &desc_entries->desc[i], data);

            if (ret < 0) {

                goto exit;

            }

        }

        if (bdrv_getlength(bs->file->bs) < desc_entries->hdr.last_file_offset) {

            new_file_size = desc_entries->hdr.last_file_offset;

            if (new_file_size % (1024*1024)) {

                /* round up to nearest 1MB boundary */

                new_file_size = ((new_file_size >> 20) + 1) << 20;

                bdrv_truncate(bs->file, new_file_size, PREALLOC_MODE_OFF, NULL);

            }

        }

        qemu_vfree(desc_entries);

        desc_entries = NULL;

    }



    bdrv_flush(bs);

    /* once the log is fully flushed, indicate that we have an empty log

     * now.  This also sets the log guid to 0, to indicate an empty log */

    vhdx_log_reset(bs, s);



exit:

    qemu_vfree(data);

    qemu_vfree(desc_entries);

    return ret;

}
