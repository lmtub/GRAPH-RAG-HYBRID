static int mov_read_cmov(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

#if CONFIG_ZLIB

    AVIOContext ctx;

    uint8_t *cmov_data;

    uint8_t *moov_data; /* uncompressed data */

    long cmov_len, moov_len;

    int ret = -1;



    avio_rb32(pb); /* dcom atom */

    if (avio_rl32(pb) != MKTAG('d','c','o','m'))

        return AVERROR_INVALIDDATA;

    if (avio_rl32(pb) != MKTAG('z','l','i','b')) {

        av_log(c->fc, AV_LOG_ERROR, "unknown compression for cmov atom !");

        return AVERROR_INVALIDDATA;

    }

    avio_rb32(pb); /* cmvd atom */

    if (avio_rl32(pb) != MKTAG('c','m','v','d'))

        return AVERROR_INVALIDDATA;

    moov_len = avio_rb32(pb); /* uncompressed size */

    cmov_len = atom.size - 6 * 4;



    cmov_data = av_malloc(cmov_len);

    if (!cmov_data)

        return AVERROR(ENOMEM);

    moov_data = av_malloc(moov_len);

    if (!moov_data) {

        av_free(cmov_data);

        return AVERROR(ENOMEM);

    }

    avio_read(pb, cmov_data, cmov_len);

    if (uncompress (moov_data, (uLongf *) &moov_len, (const Bytef *)cmov_data, cmov_len) != Z_OK)

        goto free_and_return;

    if (ffio_init_context(&ctx, moov_data, moov_len, 0, NULL, NULL, NULL, NULL) != 0)

        goto free_and_return;

    atom.type = MKTAG('m','o','o','v');

    atom.size = moov_len;

    ret = mov_read_default(c, &ctx, atom);

free_and_return:

    av_free(moov_data);

    av_free(cmov_data);

    return ret;

#else

    av_log(c->fc, AV_LOG_ERROR, "this file requires zlib support compiled in\n");

    return AVERROR(ENOSYS);

#endif

}
