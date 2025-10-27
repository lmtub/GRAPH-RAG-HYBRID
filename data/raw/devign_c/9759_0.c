static int decode_extradata_ps(const uint8_t *data, int size, H264ParamSets *ps,

                               int is_avc, void *logctx)

{

    H2645Packet pkt = { 0 };

    int i, ret = 0;



    ret = ff_h2645_packet_split(&pkt, data, size, logctx, is_avc, 2, AV_CODEC_ID_H264);

    if (ret < 0) {

        ret = 0;

        goto fail;

    }



    for (i = 0; i < pkt.nb_nals; i++) {

        H2645NAL *nal = &pkt.nals[i];

        switch (nal->type) {

        case H264_NAL_SPS:

            ret = ff_h264_decode_seq_parameter_set(&nal->gb, logctx, ps, 0);

            if (ret < 0)

                goto fail;

            break;

        case H264_NAL_PPS:

            ret = ff_h264_decode_picture_parameter_set(&nal->gb, logctx, ps,

                                                       nal->size_bits);

            if (ret < 0)

                goto fail;

            break;

        default:

            av_log(logctx, AV_LOG_VERBOSE, "Ignoring NAL type %d in extradata\n",

                   nal->type);

            break;

        }

    }



fail:

    ff_h2645_packet_uninit(&pkt);

    return ret;

}
