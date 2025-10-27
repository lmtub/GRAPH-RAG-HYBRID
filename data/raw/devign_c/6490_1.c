static int run_test(AVCodec *enc, AVCodec *dec, AVCodecContext *enc_ctx,

                    AVCodecContext *dec_ctx)

{

    AVPacket enc_pkt;

    AVFrame *in_frame, *out_frame;

    uint8_t *raw_in = NULL, *raw_out = NULL;

    int in_offset = 0, out_offset = 0;

    int frame_data_size = 0;

    int result = 0;

    int got_output = 0;

    int i = 0;



    in_frame = av_frame_alloc();

    if (!in_frame) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate input frame\n");

        return AVERROR(ENOMEM);

    }



    in_frame->nb_samples = enc_ctx->frame_size;

    in_frame->format = enc_ctx->sample_fmt;

    in_frame->channel_layout = enc_ctx->channel_layout;

    if (av_frame_get_buffer(in_frame, 32) != 0) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate a buffer for input frame\n");

        return AVERROR(ENOMEM);

    }



    out_frame = av_frame_alloc();

    if (!out_frame) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate output frame\n");

        return AVERROR(ENOMEM);

    }



    raw_in = av_malloc(in_frame->linesize[0] * NUMBER_OF_FRAMES);

    if (!raw_in) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate memory for raw_in\n");

        return AVERROR(ENOMEM);

    }



    raw_out = av_malloc(in_frame->linesize[0] * NUMBER_OF_FRAMES);

    if (!raw_out) {

        av_log(NULL, AV_LOG_ERROR, "Can't allocate memory for raw_out\n");

        return AVERROR(ENOMEM);

    }



    for (i = 0; i < NUMBER_OF_FRAMES; i++) {

        av_init_packet(&enc_pkt);

        enc_pkt.data = NULL;

        enc_pkt.size = 0;



        generate_raw_frame((uint16_t*)(in_frame->data[0]), i, enc_ctx->sample_rate,

                           enc_ctx->channels, enc_ctx->frame_size);

        memcpy(raw_in + in_offset, in_frame->data[0], in_frame->linesize[0]);

        in_offset += in_frame->linesize[0];

        result = avcodec_encode_audio2(enc_ctx, &enc_pkt, in_frame, &got_output);

        if (result < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error encoding audio frame\n");

            return result;

        }



        /* if we get an encoded packet, feed it straight to the decoder */

        if (got_output) {

            result = avcodec_decode_audio4(dec_ctx, out_frame, &got_output, &enc_pkt);

            if (result < 0) {

                av_log(NULL, AV_LOG_ERROR, "Error decoding audio packet\n");

                return result;

            }



            if (got_output) {

                if (result != enc_pkt.size) {

                    av_log(NULL, AV_LOG_INFO, "Decoder consumed only part of a packet, it is allowed to do so -- need to update this test\n");

                    return AVERROR_UNKNOWN;

                }



                if (in_frame->nb_samples != out_frame->nb_samples) {

                    av_log(NULL, AV_LOG_ERROR, "Error frames before and after decoding has different number of samples\n");

                    return AVERROR_UNKNOWN;

                }



                if (in_frame->channel_layout != out_frame->channel_layout) {

                    av_log(NULL, AV_LOG_ERROR, "Error frames before and after decoding has different channel layout\n");

                    return AVERROR_UNKNOWN;

                }



                if (in_frame->format != out_frame->format) {

                    av_log(NULL, AV_LOG_ERROR, "Error frames before and after decoding has different sample format\n");

                    return AVERROR_UNKNOWN;

                }

                memcpy(raw_out + out_offset, out_frame->data[0], out_frame->linesize[0]);

                out_offset += out_frame->linesize[0];

            }

        }

        av_free_packet(&enc_pkt);

    }



    if (memcmp(raw_in, raw_out, frame_data_size * NUMBER_OF_FRAMES) != 0) {

        av_log(NULL, AV_LOG_ERROR, "Output differs\n");

        return 1;

    }



    av_log(NULL, AV_LOG_INFO, "OK\n");



    av_freep(&raw_in);

    av_freep(&raw_out);

    av_frame_free(&in_frame);

    av_frame_free(&out_frame);

    return 0;

}
