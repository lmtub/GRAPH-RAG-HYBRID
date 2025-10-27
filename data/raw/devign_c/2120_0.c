int ff_audio_convert(AudioConvert *ac, AudioData *out, AudioData *in)

{

    int use_generic = 1;

    int len         = in->nb_samples;

    int p;



    if (ac->dc) {

        /* dithered conversion */

        av_dlog(ac->avr, "%d samples - audio_convert: %s to %s (dithered)\n",

                len, av_get_sample_fmt_name(ac->in_fmt),

                av_get_sample_fmt_name(ac->out_fmt));



        return ff_convert_dither(ac->dc, out, in);

    }



    /* determine whether to use the optimized function based on pointer and

       samples alignment in both the input and output */

    if (ac->has_optimized_func) {

        int ptr_align     = FFMIN(in->ptr_align,     out->ptr_align);

        int samples_align = FFMIN(in->samples_align, out->samples_align);

        int aligned_len   = FFALIGN(len, ac->samples_align);

        if (!(ptr_align % ac->ptr_align) && samples_align >= aligned_len) {

            len = aligned_len;

            use_generic = 0;

        }

    }

    av_dlog(ac->avr, "%d samples - audio_convert: %s to %s (%s)\n", len,

            av_get_sample_fmt_name(ac->in_fmt),

            av_get_sample_fmt_name(ac->out_fmt),

            use_generic ? ac->func_descr_generic : ac->func_descr);



    if (ac->apply_map) {

        ChannelMapInfo *map = &ac->avr->ch_map_info;



        if (!av_sample_fmt_is_planar(ac->out_fmt)) {

            av_log(ac->avr, AV_LOG_ERROR, "cannot remap packed format during conversion\n");

            return AVERROR(EINVAL);

        }



        if (map->do_remap) {

            if (av_sample_fmt_is_planar(ac->in_fmt)) {

                conv_func_flat *convert = use_generic ? ac->conv_flat_generic :

                                                        ac->conv_flat;



                for (p = 0; p < ac->planes; p++)

                    if (map->channel_map[p] >= 0)

                        convert(out->data[p], in->data[map->channel_map[p]], len);

            } else {

                uint8_t *data[AVRESAMPLE_MAX_CHANNELS];

                conv_func_deinterleave *convert = use_generic ?

                                                  ac->conv_deinterleave_generic :

                                                  ac->conv_deinterleave;



                for (p = 0; p < ac->channels; p++)

                    data[map->input_map[p]] = out->data[p];



                convert(data, in->data[0], len, ac->channels);

            }

        }

        if (map->do_copy || map->do_zero) {

            for (p = 0; p < ac->planes; p++) {

                if (map->channel_copy[p])

                    memcpy(out->data[p], out->data[map->channel_copy[p]],

                           len * out->stride);

                else if (map->channel_zero[p])

                    av_samples_set_silence(&out->data[p], 0, len, 1, ac->out_fmt);

            }

        }

    } else {

        switch (ac->func_type) {

        case CONV_FUNC_TYPE_FLAT: {

            if (!in->is_planar)

                len *= in->channels;

            if (use_generic) {

                for (p = 0; p < ac->planes; p++)

                    ac->conv_flat_generic(out->data[p], in->data[p], len);

            } else {

                for (p = 0; p < ac->planes; p++)

                    ac->conv_flat(out->data[p], in->data[p], len);

            }

            break;

        }

        case CONV_FUNC_TYPE_INTERLEAVE:

            if (use_generic)

                ac->conv_interleave_generic(out->data[0], in->data, len,

                                            ac->channels);

            else

                ac->conv_interleave(out->data[0], in->data, len, ac->channels);

            break;

        case CONV_FUNC_TYPE_DEINTERLEAVE:

            if (use_generic)

                ac->conv_deinterleave_generic(out->data, in->data[0], len,

                                              ac->channels);

            else

                ac->conv_deinterleave(out->data, in->data[0], len,

                                      ac->channels);

            break;

        }

    }



    out->nb_samples = in->nb_samples;

    return 0;

}
