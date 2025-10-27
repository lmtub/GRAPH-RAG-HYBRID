static int mxf_read_primer_pack(MXFContext *mxf)

{

    ByteIOContext *pb = mxf->fc->pb;

    int item_num = get_be32(pb);

    int item_len = get_be32(pb);



    if (item_len != 18) {

        av_log(mxf->fc, AV_LOG_ERROR, "unsupported primer pack item length\n");

        return -1;

    }

    if (item_num > UINT_MAX / item_len)

        return -1;

    mxf->local_tags_count = item_num;

    mxf->local_tags = av_malloc(item_num*item_len);

    if (!mxf->local_tags)

        return -1;

    get_buffer(pb, mxf->local_tags, item_num*item_len);

    return 0;

}
