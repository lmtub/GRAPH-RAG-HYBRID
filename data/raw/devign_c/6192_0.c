int ff_ivi_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                        AVPacket *avpkt)

{

    IVI45DecContext *ctx = avctx->priv_data;

    const uint8_t   *buf = avpkt->data;

    AVFrame       *frame = data;

    int             buf_size = avpkt->size;

    int             result, p, b;



    init_get_bits(&ctx->gb, buf, buf_size * 8);

    ctx->frame_data = buf;

    ctx->frame_size = buf_size;



    result = ctx->decode_pic_hdr(ctx, avctx);

    if (result) {

        av_log(avctx, AV_LOG_ERROR,

               "Error while decoding picture header: %d\n", result);

        return result;

    }

    if (ctx->gop_invalid)

        return AVERROR_INVALIDDATA;



    if (avctx->codec_id == AV_CODEC_ID_INDEO4 &&

        ctx->frame_type == IVI4_FRAMETYPE_NULL_LAST) {

        if (ctx->got_p_frame) {

            av_frame_move_ref(data, ctx->p_frame);

            *got_frame = 1;

            ctx->got_p_frame = 0;

        } else {

            *got_frame = 0;

        }

        return buf_size;

    }



    if (ctx->gop_flags & IVI5_IS_PROTECTED) {

        avpriv_report_missing_feature(avctx, "Password-protected clip!\n");

        return AVERROR_PATCHWELCOME;

    }



    if (!ctx->planes[0].bands) {

        av_log(avctx, AV_LOG_ERROR, "Color planes not initialized yet\n");

        return AVERROR_INVALIDDATA;

    }



    ctx->switch_buffers(ctx);



    //{ START_TIMER;



    if (ctx->is_nonnull_frame(ctx)) {

        for (p = 0; p < 3; p++) {

            for (b = 0; b < ctx->planes[p].num_bands; b++) {

                result = decode_band(ctx, &ctx->planes[p].bands[b], avctx);

                if (result < 0) {

                    av_log(avctx, AV_LOG_ERROR,

                           "Error while decoding band: %d, plane: %d\n", b, p);

                    return result;

                }

            }

        }

    } else {

        if (ctx->is_scalable)

            return AVERROR_INVALIDDATA;



        for (p = 0; p < 3; p++) {

            if (!ctx->planes[p].bands[0].buf)

                return AVERROR_INVALIDDATA;

        }

    }



    //STOP_TIMER("decode_planes"); }



    result = ff_set_dimensions(avctx, ctx->planes[0].width, ctx->planes[0].height);

    if (result < 0)

        return result;



    if ((result = ff_get_buffer(avctx, frame, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return result;

    }



    if (ctx->is_scalable) {

        if (avctx->codec_id == AV_CODEC_ID_INDEO4)

            ff_ivi_recompose_haar(&ctx->planes[0], frame->data[0], frame->linesize[0]);

        else

            ff_ivi_recompose53   (&ctx->planes[0], frame->data[0], frame->linesize[0]);

    } else {

        ivi_output_plane(&ctx->planes[0], frame->data[0], frame->linesize[0]);

    }



    ivi_output_plane(&ctx->planes[2], frame->data[1], frame->linesize[1]);

    ivi_output_plane(&ctx->planes[1], frame->data[2], frame->linesize[2]);



    *got_frame = 1;



    /* If the bidirectional mode is enabled, next I and the following P

     * frame will be sent together. Unfortunately the approach below seems

     * to be the only way to handle the B-frames mode.

     * That's exactly the same Intel decoders do.

     */

    if (avctx->codec_id == AV_CODEC_ID_INDEO4 &&

        ctx->frame_type == IVI4_FRAMETYPE_INTRA) {

        int left;



        while (get_bits(&ctx->gb, 8)); // skip version string

        left = get_bits_count(&ctx->gb) & 0x18;

        skip_bits_long(&ctx->gb, 64 - left);

        if (get_bits_left(&ctx->gb) > 18 &&

            show_bits_long(&ctx->gb, 21) == 0xBFFF8) { // syncheader + inter type

            AVPacket pkt;

            pkt.data = avpkt->data + (get_bits_count(&ctx->gb) >> 3);

            pkt.size = get_bits_left(&ctx->gb) >> 3;

            ff_ivi_decode_frame(avctx, ctx->p_frame, &ctx->got_p_frame, &pkt);

        }

    }



    return buf_size;

}
