static int aac_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame_ptr, AVPacket *avpkt)

{

    AACContext *ac = avctx->priv_data;

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    GetBitContext gb;

    int buf_consumed;

    int buf_offset;

    int err;

    int new_extradata_size;

    const uint8_t *new_extradata = av_packet_get_side_data(avpkt,

                                       AV_PKT_DATA_NEW_EXTRADATA,

                                       &new_extradata_size);



    if (new_extradata) {

        av_free(avctx->extradata);

        avctx->extradata = av_mallocz(new_extradata_size +

                                      FF_INPUT_BUFFER_PADDING_SIZE);

        if (!avctx->extradata)

            return AVERROR(ENOMEM);

        avctx->extradata_size = new_extradata_size;

        memcpy(avctx->extradata, new_extradata, new_extradata_size);

        push_output_configuration(ac);

        if (decode_audio_specific_config(ac, ac->avctx, &ac->oc[1].m4ac,

                                         avctx->extradata,

                                         avctx->extradata_size*8, 1) < 0) {

            pop_output_configuration(ac);

            return AVERROR_INVALIDDATA;

        }

    }



    init_get_bits(&gb, buf, buf_size * 8);



    if ((err = aac_decode_frame_int(avctx, data, got_frame_ptr, &gb)) < 0)

        return err;



    buf_consumed = (get_bits_count(&gb) + 7) >> 3;

    for (buf_offset = buf_consumed; buf_offset < buf_size; buf_offset++)

        if (buf[buf_offset])

            break;



    return buf_size > buf_offset ? buf_consumed : buf_size;

}
