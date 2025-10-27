int vhdx_log_write_and_flush(BlockDriverState *bs, BDRVVHDXState *s,

                             void *data, uint32_t length, uint64_t offset)

{

    int ret = 0;

    VHDXLogSequence logs = { .valid = true,

                             .count = 1,

                             .hdr = { 0 } };





    /* Make sure data written (new and/or changed blocks) is stable

     * on disk, before creating log entry */

    bdrv_flush(bs);

    ret = vhdx_log_write(bs, s, data, length, offset);

    if (ret < 0) {

        goto exit;

    }

    logs.log = s->log;



    /* Make sure log is stable on disk */

    bdrv_flush(bs);

    ret = vhdx_log_flush(bs, s, &logs);

    if (ret < 0) {

        goto exit;

    }



    s->log = logs.log;



exit:

    return ret;

}
