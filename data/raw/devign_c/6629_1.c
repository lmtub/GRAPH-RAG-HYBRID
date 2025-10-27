static void get_private_data(OutputStream *os)

{

    AVCodecContext *codec = os->ctx->streams[0]->codec;

    uint8_t *ptr = codec->extradata;

    int size = codec->extradata_size;

    int i;

    if (codec->codec_id == AV_CODEC_ID_H264) {

        ff_avc_write_annexb_extradata(ptr, &ptr, &size);

        if (!ptr)

            ptr = codec->extradata;

    }

    if (!ptr)

        return;

    os->private_str = av_mallocz(2*size + 1);

    if (!os->private_str)

        return;

    for (i = 0; i < size; i++)

        snprintf(&os->private_str[2*i], 3, "%02x", ptr[i]);

    if (ptr != codec->extradata)

        av_free(ptr);

}
