static int vhdx_parse_log(BlockDriverState *bs, BDRVVHDXState *s)

{

    int ret = 0;

    int i;

    VHDXHeader *hdr;



    hdr = s->headers[s->curr_header];



    /* either the log guid, or log length is zero,

     * then a replay log is present */

    for (i = 0; i < sizeof(hdr->log_guid.data4); i++) {

        ret |= hdr->log_guid.data4[i];

    }

    if (hdr->log_guid.data1 == 0 &&

        hdr->log_guid.data2 == 0 &&

        hdr->log_guid.data3 == 0 &&

        ret == 0) {

        goto exit;

    }



    /* per spec, only log version of 0 is supported */

    if (hdr->log_version != 0) {

        ret = -EINVAL;

        goto exit;

    }



    if (hdr->log_length == 0) {

        goto exit;

    }



    /* We currently do not support images with logs to replay */

    ret = -ENOTSUP;



exit:

    return ret;

}
