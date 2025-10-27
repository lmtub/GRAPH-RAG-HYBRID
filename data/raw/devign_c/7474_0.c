static int roq_dpcm_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                                 const AVFrame *frame, int *got_packet_ptr)

{

    int i, stereo, data_size, ret;

    const int16_t *in = frame ? (const int16_t *)frame->data[0] : NULL;

    uint8_t *out;

    ROQDPCMContext *context = avctx->priv_data;



    stereo = (avctx->channels == 2);



    if (!in && context->input_frames >= 8)

        return 0;



    if (in && context->input_frames < 8) {

        memcpy(&context->frame_buffer[context->buffered_samples * avctx->channels],

               in, avctx->frame_size * avctx->channels * sizeof(*in));

        context->buffered_samples += avctx->frame_size;

        if (context->input_frames == 0)

            context->first_pts = frame->pts;

        if (context->input_frames < 7) {

            context->input_frames++;

            return 0;

        }

    }

    if (context->input_frames < 8) {

        in = context->frame_buffer;

    }



    if (stereo) {

        context->lastSample[0] &= 0xFF00;

        context->lastSample[1] &= 0xFF00;

    }



    if (context->input_frames == 7)

        data_size = avctx->channels * context->buffered_samples;

    else

        data_size = avctx->channels * avctx->frame_size;



    if ((ret = ff_alloc_packet2(avctx, avpkt, ROQ_HEADER_SIZE + data_size)))

        return ret;

    out = avpkt->data;



    bytestream_put_byte(&out, stereo ? 0x21 : 0x20);

    bytestream_put_byte(&out, 0x10);

    bytestream_put_le32(&out, data_size);



    if (stereo) {

        bytestream_put_byte(&out, (context->lastSample[1])>>8);

        bytestream_put_byte(&out, (context->lastSample[0])>>8);

    } else

        bytestream_put_le16(&out, context->lastSample[0]);



    /* Write the actual samples */

    for (i = 0; i < data_size; i++)

        *out++ = dpcm_predict(&context->lastSample[i & 1], *in++);



    avpkt->pts      = context->input_frames <= 7 ? context->first_pts : frame->pts;

    avpkt->duration = data_size / avctx->channels;



    context->input_frames++;

    if (!in)

        context->input_frames = FFMAX(context->input_frames, 8);



    *got_packet_ptr = 1;

    return 0;

}
