static av_cold int qdm2_decode_init(AVCodecContext *avctx)

{

    QDM2Context *s = avctx->priv_data;

    uint8_t *extradata;

    int extradata_size;

    int tmp_val, tmp, size;



    /* extradata parsing



    Structure:

    wave {

        frma (QDM2)

        QDCA

        QDCP

    }



    32  size (including this field)

    32  tag (=frma)

    32  type (=QDM2 or QDMC)



    32  size (including this field, in bytes)

    32  tag (=QDCA) // maybe mandatory parameters

    32  unknown (=1)

    32  channels (=2)

    32  samplerate (=44100)

    32  bitrate (=96000)

    32  block size (=4096)

    32  frame size (=256) (for one channel)

    32  packet size (=1300)



    32  size (including this field, in bytes)

    32  tag (=QDCP) // maybe some tuneable parameters

    32  float1 (=1.0)

    32  zero ?

    32  float2 (=1.0)

    32  float3 (=1.0)

    32  unknown (27)

    32  unknown (8)

    32  zero ?

    */



    if (!avctx->extradata || (avctx->extradata_size < 48)) {

        av_log(avctx, AV_LOG_ERROR, "extradata missing or truncated\n");

        return -1;

    }



    extradata = avctx->extradata;

    extradata_size = avctx->extradata_size;



    while (extradata_size > 7) {

        if (!memcmp(extradata, "frmaQDM", 7))

            break;

        extradata++;

        extradata_size--;

    }



    if (extradata_size < 12) {

        av_log(avctx, AV_LOG_ERROR, "not enough extradata (%i)\n",

               extradata_size);

        return -1;

    }



    if (memcmp(extradata, "frmaQDM", 7)) {

        av_log(avctx, AV_LOG_ERROR, "invalid headers, QDM? not found\n");

        return -1;

    }



    if (extradata[7] == 'C') {

//        s->is_qdmc = 1;

        av_log(avctx, AV_LOG_ERROR, "stream is QDMC version 1, which is not supported\n");

        return -1;

    }



    extradata += 8;

    extradata_size -= 8;



    size = AV_RB32(extradata);



    if(size > extradata_size){

        av_log(avctx, AV_LOG_ERROR, "extradata size too small, %i < %i\n",

               extradata_size, size);

        return -1;

    }



    extradata += 4;

    av_log(avctx, AV_LOG_DEBUG, "size: %d\n", size);

    if (AV_RB32(extradata) != MKBETAG('Q','D','C','A')) {

        av_log(avctx, AV_LOG_ERROR, "invalid extradata, expecting QDCA\n");

        return -1;

    }



    extradata += 8;



    avctx->channels = s->nb_channels = s->channels = AV_RB32(extradata);

    extradata += 4;

    if (s->channels > MPA_MAX_CHANNELS)

        return AVERROR_INVALIDDATA;



    avctx->sample_rate = AV_RB32(extradata);

    extradata += 4;



    avctx->bit_rate = AV_RB32(extradata);

    extradata += 4;



    s->group_size = AV_RB32(extradata);

    extradata += 4;



    s->fft_size = AV_RB32(extradata);

    extradata += 4;



    s->checksum_size = AV_RB32(extradata);

    if (s->checksum_size >= 1U << 28) {

        av_log(avctx, AV_LOG_ERROR, "data block size too large (%u)\n", s->checksum_size);

        return AVERROR_INVALIDDATA;

    }



    s->fft_order = av_log2(s->fft_size) + 1;

    s->fft_frame_size = 2 * s->fft_size; // complex has two floats



    // something like max decodable tones

    s->group_order = av_log2(s->group_size) + 1;

    s->frame_size = s->group_size / 16; // 16 iterations per super block

    if (s->frame_size > QDM2_MAX_FRAME_SIZE)

        return AVERROR_INVALIDDATA;



    s->sub_sampling = s->fft_order - 7;

    s->frequency_range = 255 / (1 << (2 - s->sub_sampling));



    switch ((s->sub_sampling * 2 + s->channels - 1)) {

        case 0: tmp = 40; break;

        case 1: tmp = 48; break;

        case 2: tmp = 56; break;

        case 3: tmp = 72; break;

        case 4: tmp = 80; break;

        case 5: tmp = 100;break;

        default: tmp=s->sub_sampling; break;

    }

    tmp_val = 0;

    if ((tmp * 1000) < avctx->bit_rate)  tmp_val = 1;

    if ((tmp * 1440) < avctx->bit_rate)  tmp_val = 2;

    if ((tmp * 1760) < avctx->bit_rate)  tmp_val = 3;

    if ((tmp * 2240) < avctx->bit_rate)  tmp_val = 4;

    s->cm_table_select = tmp_val;



    if (s->sub_sampling == 0)

        tmp = 7999;

    else

        tmp = ((-(s->sub_sampling -1)) & 8000) + 20000;

    /*

    0: 7999 -> 0

    1: 20000 -> 2

    2: 28000 -> 2

    */

    if (tmp < 8000)

        s->coeff_per_sb_select = 0;

    else if (tmp <= 16000)

        s->coeff_per_sb_select = 1;

    else

        s->coeff_per_sb_select = 2;



    // Fail on unknown fft order

    if ((s->fft_order < 7) || (s->fft_order > 9)) {

        av_log(avctx, AV_LOG_ERROR, "Unknown FFT order (%d), contact the developers!\n", s->fft_order);

        return -1;

    }



    ff_rdft_init(&s->rdft_ctx, s->fft_order, IDFT_C2R);

    ff_mpadsp_init(&s->mpadsp);



    qdm2_init(s);



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    avcodec_get_frame_defaults(&s->frame);

    avctx->coded_frame = &s->frame;



//    dump_context(s);

    return 0;

}
