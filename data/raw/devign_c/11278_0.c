int AC3_NAME(encode_frame)(AVCodecContext *avctx, AVPacket *avpkt,

                           const AVFrame *frame, int *got_packet_ptr)

{

    AC3EncodeContext *s = avctx->priv_data;

    int ret;



    if (s->options.allow_per_frame_metadata) {

        ret = ff_ac3_validate_metadata(s);

        if (ret)

            return ret;

    }



    if (s->bit_alloc.sr_code == 1 || s->eac3)

        ff_ac3_adjust_frame_size(s);



    copy_input_samples(s, (SampleType **)frame->extended_data);



    apply_mdct(s);



    if (s->fixed_point)

        scale_coefficients(s);



    clip_coefficients(&s->dsp, s->blocks[0].mdct_coef[1],

                      AC3_MAX_COEFS * s->num_blocks * s->channels);



    s->cpl_on = s->cpl_enabled;

    ff_ac3_compute_coupling_strategy(s);



    if (s->cpl_on)

        apply_channel_coupling(s);



    compute_rematrixing_strategy(s);



    if (!s->fixed_point)

        scale_coefficients(s);



    ff_ac3_apply_rematrixing(s);



    ff_ac3_process_exponents(s);



    ret = ff_ac3_compute_bit_allocation(s);

    if (ret) {

        av_log(avctx, AV_LOG_ERROR, "Bit allocation failed. Try increasing the bitrate.\n");

        return ret;

    }



    ff_ac3_group_exponents(s);



    ff_ac3_quantize_mantissas(s);



    if ((ret = ff_alloc_packet2(avctx, avpkt, s->frame_size)))

        return ret;

    ff_ac3_output_frame(s, avpkt->data);



    if (frame->pts != AV_NOPTS_VALUE)

        avpkt->pts = frame->pts - ff_samples_to_time_base(avctx, avctx->delay);



    *got_packet_ptr = 1;

    return 0;

}
