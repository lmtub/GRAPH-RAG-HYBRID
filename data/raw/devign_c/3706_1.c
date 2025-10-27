static int s302m_encode2_frame(AVCodecContext *avctx, AVPacket *avpkt,

                               const AVFrame *frame, int *got_packet_ptr)

{

    S302MEncContext *s = avctx->priv_data;

    const int buf_size = AES3_HEADER_LEN +

                        (frame->nb_samples *

                         avctx->channels *

                        (avctx->bits_per_raw_sample + 4)) / 8;

    int ret, c, channels;

    uint8_t *o;

    PutBitContext pb;



    if ((ret = ff_alloc_packet2(avctx, avpkt, buf_size)) < 0)

        return ret;



    o = avpkt->data;

    init_put_bits(&pb, o, buf_size * 8);

    put_bits(&pb, 16, buf_size - AES3_HEADER_LEN);

    put_bits(&pb, 2, (avctx->channels - 2) >> 1);   // number of channels

    put_bits(&pb, 8, 0);                            // channel ID

    put_bits(&pb, 2, (avctx->bits_per_raw_sample - 16) / 4); // bits per samples (0 = 16bit, 1 = 20bit, 2 = 24bit)

    put_bits(&pb, 4, 0);                            // alignments

    flush_put_bits(&pb);

    o += AES3_HEADER_LEN;



    if (avctx->bits_per_raw_sample == 24) {

        const uint32_t *samples = (uint32_t *)frame->data[0];



        for (c = 0; c < frame->nb_samples; c++) {

            uint8_t vucf = s->framing_index == 0 ? 0x10: 0;



            for (channels = 0; channels < avctx->channels; channels += 2) {

                o[0] = ff_reverse[(samples[0] & 0x0000FF00) >> 8];

                o[1] = ff_reverse[(samples[0] & 0x00FF0000) >> 16];

                o[2] = ff_reverse[(samples[0] & 0xFF000000) >> 24];

                o[3] = ff_reverse[(samples[1] & 0x00000F00) >> 4] | vucf;

                o[4] = ff_reverse[(samples[1] & 0x000FF000) >> 12];

                o[5] = ff_reverse[(samples[1] & 0x0FF00000) >> 20];

                o[6] = ff_reverse[(samples[1] & 0xF0000000) >> 28];

                o += 7;

                samples += 2;

            }



            s->framing_index++;

            if (s->framing_index >= 192)

                s->framing_index = 0;

        }

    } else if (avctx->bits_per_raw_sample == 20) {

        const uint32_t *samples = (uint32_t *)frame->data[0];



        for (c = 0; c < frame->nb_samples; c++) {

            uint8_t vucf = s->framing_index == 0 ? 0x80: 0;



            for (channels = 0; channels < avctx->channels; channels += 2) {

                o[0] = ff_reverse[ (samples[0] & 0x000FF000) >> 12];

                o[1] = ff_reverse[ (samples[0] & 0x0FF00000) >> 20];

                o[2] = ff_reverse[((samples[0] & 0xF0000000) >> 28) | vucf];

                o[3] = ff_reverse[ (samples[1] & 0x000FF000) >> 12];

                o[4] = ff_reverse[ (samples[1] & 0x0FF00000) >> 20];

                o[5] = ff_reverse[ (samples[1] & 0xF0000000) >> 28];

                o += 6;

                samples += 2;

            }



            s->framing_index++;

            if (s->framing_index >= 192)

                s->framing_index = 0;

        }

    } else if (avctx->bits_per_raw_sample == 16) {

        const uint16_t *samples = (uint16_t *)frame->data[0];



        for (c = 0; c < frame->nb_samples; c++) {

            uint8_t vucf = s->framing_index == 0 ? 0x10 : 0;



            for (channels = 0; channels < avctx->channels; channels += 2) {

                o[0] = ff_reverse[ samples[0] & 0xFF];

                o[1] = ff_reverse[(samples[0] & 0xFF00) >>  8];

                o[2] = ff_reverse[(samples[1] & 0x0F)   <<  4] | vucf;

                o[3] = ff_reverse[(samples[1] & 0x0FF0) >>  4];

                o[4] = ff_reverse[(samples[1] & 0xF000) >> 12];

                o += 5;

                samples += 2;



            }



            s->framing_index++;

            if (s->framing_index >= 192)

                s->framing_index = 0;

        }

    }



    *got_packet_ptr = 1;



    return 0;

}
