static int mxf_decrypt_triplet(AVFormatContext *s, AVPacket *pkt, KLVPacket *klv)

{

    static const uint8_t checkv[16] = {0x43, 0x48, 0x55, 0x4b, 0x43, 0x48, 0x55, 0x4b, 0x43, 0x48, 0x55, 0x4b, 0x43, 0x48, 0x55, 0x4b};

    MXFContext *mxf = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t end = avio_tell(pb) + klv->length;

    uint64_t size;

    uint64_t orig_size;

    uint64_t plaintext_size;

    uint8_t ivec[16];

    uint8_t tmpbuf[16];

    int index;



    if (!mxf->aesc && s->key && s->keylen == 16) {

        mxf->aesc = av_malloc(av_aes_size);

        if (!mxf->aesc)

            return -1;

        av_aes_init(mxf->aesc, s->key, 128, 1);

    }

    // crypto context

    avio_skip(pb, klv_decode_ber_length(pb));

    // plaintext offset

    klv_decode_ber_length(pb);

    plaintext_size = avio_rb64(pb);

    // source klv key

    klv_decode_ber_length(pb);

    avio_read(pb, klv->key, 16);

    if (!IS_KLV_KEY(klv, mxf_essence_element_key))

        return -1;

    index = mxf_get_stream_index(s, klv);

    if (index < 0)

        return -1;

    // source size

    klv_decode_ber_length(pb);

    orig_size = avio_rb64(pb);

    if (orig_size < plaintext_size)

        return -1;

    // enc. code

    size = klv_decode_ber_length(pb);

    if (size < 32 || size - 32 < orig_size)

        return -1;

    avio_read(pb, ivec, 16);

    avio_read(pb, tmpbuf, 16);

    if (mxf->aesc)

        av_aes_crypt(mxf->aesc, tmpbuf, tmpbuf, 1, ivec, 1);

    if (memcmp(tmpbuf, checkv, 16))

        av_log(s, AV_LOG_ERROR, "probably incorrect decryption key\n");

    size -= 32;

    size = av_get_packet(pb, pkt, size);

    if (size < 0)

        return size;

    else if (size < plaintext_size)

        return AVERROR_INVALIDDATA;

    size -= plaintext_size;

    if (mxf->aesc)

        av_aes_crypt(mxf->aesc, &pkt->data[plaintext_size],

                     &pkt->data[plaintext_size], size >> 4, ivec, 1);

    av_shrink_packet(pkt, orig_size);

    pkt->stream_index = index;

    avio_skip(pb, end - avio_tell(pb));

    return 0;

}
