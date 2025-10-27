static int alac_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame_ptr, AVPacket *avpkt)

{

    ALACContext *alac = avctx->priv_data;

    enum RawDataBlockType element;

    int channels;

    int ch, ret, got_end;



    init_get_bits(&alac->gb, avpkt->data, avpkt->size * 8);



    got_end = 0;

    alac->nb_samples = 0;

    ch = 0;

    while (get_bits_left(&alac->gb) >= 3) {

        element = get_bits(&alac->gb, 3);

        if (element == TYPE_END) {

            got_end = 1;

            break;

        }

        if (element > TYPE_CPE && element != TYPE_LFE) {

            av_log(avctx, AV_LOG_ERROR, "syntax element unsupported: %d\n", element);

            return AVERROR_PATCHWELCOME;

        }



        channels = (element == TYPE_CPE) ? 2 : 1;

        if (ch + channels > alac->channels) {

            av_log(avctx, AV_LOG_ERROR, "invalid element channel count\n");

            return AVERROR_INVALIDDATA;

        }



        ret = decode_element(avctx, data,

                             alac_channel_layout_offsets[alac->channels - 1][ch],

                             channels);

        if (ret < 0 && get_bits_left(&alac->gb))

            return ret;



        ch += channels;

    }

    if (!got_end) {

        av_log(avctx, AV_LOG_ERROR, "no end tag found. incomplete packet.\n");

        return AVERROR_INVALIDDATA;

    }



    if (avpkt->size * 8 - get_bits_count(&alac->gb) > 8) {

        av_log(avctx, AV_LOG_ERROR, "Error : %d bits left\n",

               avpkt->size * 8 - get_bits_count(&alac->gb));

    }



    *got_frame_ptr   = 1;

    *(AVFrame *)data = alac->frame;



    return avpkt->size;

}
