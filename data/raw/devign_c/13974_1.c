static int wavpack_encode_block(WavPackEncodeContext *s,

                                int32_t *samples_l, int32_t *samples_r,

                                uint8_t *out, int out_size)

{

    int block_size, start, end, data_size, tcount, temp, m = 0;

    int i, j, ret, got_extra = 0, nb_samples = s->block_samples;

    uint32_t crc = 0xffffffffu;

    struct Decorr *dpp;

    PutByteContext pb;



    if (!(s->flags & WV_MONO) && s->optimize_mono) {

        int32_t lor = 0, diff = 0;



        for (i = 0; i < nb_samples; i++) {

            lor  |= samples_l[i] | samples_r[i];

            diff |= samples_l[i] - samples_r[i];



            if (lor && diff)

                break;

        }



        if (i == nb_samples && lor && !diff) {

            s->flags &= ~(WV_JOINT_STEREO | WV_CROSS_DECORR);

            s->flags |= WV_FALSE_STEREO;



            if (!s->false_stereo) {

                s->false_stereo = 1;

                s->num_terms = 0;

                CLEAR(s->w);

            }

        } else if (s->false_stereo) {

            s->false_stereo = 0;

            s->num_terms = 0;

            CLEAR(s->w);

        }

    }



    if (s->flags & SHIFT_MASK) {

        int shift = (s->flags & SHIFT_MASK) >> SHIFT_LSB;

        int mag = (s->flags & MAG_MASK) >> MAG_LSB;



        if (s->flags & WV_MONO_DATA)

            shift_mono(samples_l, nb_samples, shift);

        else

            shift_stereo(samples_l, samples_r, nb_samples, shift);



        if ((mag -= shift) < 0)

            s->flags &= ~MAG_MASK;

        else

            s->flags -= (1 << MAG_LSB) * shift;

    }



    if ((s->flags & WV_FLOAT_DATA) || (s->flags & MAG_MASK) >> MAG_LSB >= 24) {

        av_fast_padded_malloc(&s->orig_l, &s->orig_l_size, sizeof(int32_t) * nb_samples);

        memcpy(s->orig_l, samples_l, sizeof(int32_t) * nb_samples);

        if (!(s->flags & WV_MONO_DATA)) {

            av_fast_padded_malloc(&s->orig_r, &s->orig_r_size, sizeof(int32_t) * nb_samples);

            memcpy(s->orig_r, samples_r, sizeof(int32_t) * nb_samples);

        }



        if (s->flags & WV_FLOAT_DATA)

            got_extra = scan_float(s, samples_l, samples_r, nb_samples);

        else

            got_extra = scan_int32(s, samples_l, samples_r, nb_samples);

        s->num_terms = 0;

    } else {

        scan_int23(s, samples_l, samples_r, nb_samples);

        if (s->shift != s->int32_zeros + s->int32_ones + s->int32_dups) {

            s->shift = s->int32_zeros + s->int32_ones + s->int32_dups;

            s->num_terms = 0;

        }

    }



    if (!s->num_passes && !s->num_terms) {

        s->num_passes = 1;



        if (s->flags & WV_MONO_DATA)

            ret = wv_mono(s, samples_l, 1, 0);

        else

            ret = wv_stereo(s, samples_l, samples_r, 1, 0);



        s->num_passes = 0;

    }

    if (s->flags & WV_MONO_DATA) {

        for (i = 0; i < nb_samples; i++)

            crc += (crc << 1) + samples_l[i];



        if (s->num_passes)

            ret = wv_mono(s, samples_l, !s->num_terms, 1);

    } else {

        for (i = 0; i < nb_samples; i++)

            crc += (crc << 3) + (samples_l[i] << 1) + samples_l[i] + samples_r[i];



        if (s->num_passes)

            ret = wv_stereo(s, samples_l, samples_r, !s->num_terms, 1);

    }

    if (ret < 0)

        return ret;



    if (!s->ch_offset)

        s->flags |= WV_INITIAL_BLOCK;



    s->ch_offset += 1 + !(s->flags & WV_MONO);



    if (s->ch_offset == s->avctx->channels)

        s->flags |= WV_FINAL_BLOCK;



    bytestream2_init_writer(&pb, out, out_size);

    bytestream2_put_le32(&pb, MKTAG('w', 'v', 'p', 'k'));

    bytestream2_put_le32(&pb, 0);

    bytestream2_put_le16(&pb, 0x410);

    bytestream2_put_le16(&pb, 0);

    bytestream2_put_le32(&pb, 0);

    bytestream2_put_le32(&pb, s->sample_index);

    bytestream2_put_le32(&pb, nb_samples);

    bytestream2_put_le32(&pb, s->flags);

    bytestream2_put_le32(&pb, crc);



    if (s->flags & WV_INITIAL_BLOCK &&

        s->avctx->channel_layout != AV_CH_LAYOUT_MONO &&

        s->avctx->channel_layout != AV_CH_LAYOUT_STEREO) {

        put_metadata_block(&pb, WP_ID_CHANINFO, 5);

        bytestream2_put_byte(&pb, s->avctx->channels);

        bytestream2_put_le32(&pb, s->avctx->channel_layout);

        bytestream2_put_byte(&pb, 0);

    }



    if ((s->flags & SRATE_MASK) == SRATE_MASK) {

        put_metadata_block(&pb, WP_ID_SAMPLE_RATE, 3);

        bytestream2_put_le24(&pb, s->avctx->sample_rate);

        bytestream2_put_byte(&pb, 0);

    }



    put_metadata_block(&pb, WP_ID_DECTERMS, s->num_terms);

    for (i = 0; i < s->num_terms; i++) {

        struct Decorr *dpp = &s->decorr_passes[i];

        bytestream2_put_byte(&pb, ((dpp->value + 5) & 0x1f) | ((dpp->delta << 5) & 0xe0));

    }

    if (s->num_terms & 1)

        bytestream2_put_byte(&pb, 0);



#define WRITE_DECWEIGHT(type) do {            \

        temp = store_weight(type);    \

        bytestream2_put_byte(&pb, temp);      \

        type = restore_weight(temp);  \

    } while (0)



    bytestream2_put_byte(&pb, WP_ID_DECWEIGHTS);

    bytestream2_put_byte(&pb, 0);

    start = bytestream2_tell_p(&pb);

    for (i = s->num_terms - 1; i >= 0; --i) {

        struct Decorr *dpp = &s->decorr_passes[i];



        if (store_weight(dpp->weightA) ||

            (!(s->flags & WV_MONO_DATA) && store_weight(dpp->weightB)))

                break;

    }

    tcount = i + 1;

    for (i = 0; i < s->num_terms; i++) {

        struct Decorr *dpp = &s->decorr_passes[i];

        if (i < tcount) {

            WRITE_DECWEIGHT(dpp->weightA);

            if (!(s->flags & WV_MONO_DATA))

                WRITE_DECWEIGHT(dpp->weightB);

        } else {

            dpp->weightA = dpp->weightB = 0;

        }

    }

    end = bytestream2_tell_p(&pb);

    out[start - 2] = WP_ID_DECWEIGHTS | (((end - start) & 1) ? WP_IDF_ODD: 0);

    out[start - 1] = (end - start + 1) >> 1;

    if ((end - start) & 1)

        bytestream2_put_byte(&pb, 0);



#define WRITE_DECSAMPLE(type) do {        \

        temp = log2s(type);               \

        type = wp_exp2(temp);             \

        bytestream2_put_le16(&pb, temp);  \

    } while (0)



    bytestream2_put_byte(&pb, WP_ID_DECSAMPLES);

    bytestream2_put_byte(&pb, 0);

    start = bytestream2_tell_p(&pb);

    for (i = 0; i < s->num_terms; i++) {

        struct Decorr *dpp = &s->decorr_passes[i];

        if (i == 0) {

            if (dpp->value > MAX_TERM) {

                WRITE_DECSAMPLE(dpp->samplesA[0]);

                WRITE_DECSAMPLE(dpp->samplesA[1]);

                if (!(s->flags & WV_MONO_DATA)) {

                    WRITE_DECSAMPLE(dpp->samplesB[0]);

                    WRITE_DECSAMPLE(dpp->samplesB[1]);

                }

            } else if (dpp->value < 0) {

                WRITE_DECSAMPLE(dpp->samplesA[0]);

                WRITE_DECSAMPLE(dpp->samplesB[0]);

            } else {

                for (j = 0; j < dpp->value; j++) {

                    WRITE_DECSAMPLE(dpp->samplesA[j]);

                    if (!(s->flags & WV_MONO_DATA))

                        WRITE_DECSAMPLE(dpp->samplesB[j]);

                }

            }

        } else {

            CLEAR(dpp->samplesA);

            CLEAR(dpp->samplesB);

        }

    }

    end = bytestream2_tell_p(&pb);

    out[start - 1] = (end - start) >> 1;



#define WRITE_CHAN_ENTROPY(chan) do {               \

        for (i = 0; i < 3; i++) {                   \

            temp = wp_log2(s->w.c[chan].median[i]); \

            bytestream2_put_le16(&pb, temp);        \

            s->w.c[chan].median[i] = wp_exp2(temp); \

        }                                           \

    } while (0)



    put_metadata_block(&pb, WP_ID_ENTROPY, 6 * (1 + (!(s->flags & WV_MONO_DATA))));

    WRITE_CHAN_ENTROPY(0);

    if (!(s->flags & WV_MONO_DATA))

        WRITE_CHAN_ENTROPY(1);



    if (s->flags & WV_FLOAT_DATA) {

        put_metadata_block(&pb, WP_ID_FLOATINFO, 4);

        bytestream2_put_byte(&pb, s->float_flags);

        bytestream2_put_byte(&pb, s->float_shift);

        bytestream2_put_byte(&pb, s->float_max_exp);

        bytestream2_put_byte(&pb, 127);

    }



    if (s->flags & WV_INT32_DATA) {

        put_metadata_block(&pb, WP_ID_INT32INFO, 4);

        bytestream2_put_byte(&pb, s->int32_sent_bits);

        bytestream2_put_byte(&pb, s->int32_zeros);

        bytestream2_put_byte(&pb, s->int32_ones);

        bytestream2_put_byte(&pb, s->int32_dups);

    }



    if (s->flags & WV_MONO_DATA && !s->num_passes) {

        for (i = 0; i < nb_samples; i++) {

            int32_t code = samples_l[i];



            for (tcount = s->num_terms, dpp = s->decorr_passes; tcount--; dpp++) {

                int32_t sam;



                if (dpp->value > MAX_TERM) {

                    if (dpp->value & 1)

                        sam = 2 * dpp->samplesA[0] - dpp->samplesA[1];

                    else

                        sam = (3 * dpp->samplesA[0] - dpp->samplesA[1]) >> 1;



                    dpp->samplesA[1] = dpp->samplesA[0];

                    dpp->samplesA[0] = code;

                } else {

                    sam = dpp->samplesA[m];

                    dpp->samplesA[(m + dpp->value) & (MAX_TERM - 1)] = code;

                }



                code -= APPLY_WEIGHT(dpp->weightA, sam);

                UPDATE_WEIGHT(dpp->weightA, dpp->delta, sam, code);

            }



            m = (m + 1) & (MAX_TERM - 1);

            samples_l[i] = code;

        }

        if (m) {

            for (tcount = s->num_terms, dpp = s->decorr_passes; tcount--; dpp++)

                if (dpp->value > 0 && dpp->value <= MAX_TERM) {

                int32_t temp_A[MAX_TERM], temp_B[MAX_TERM];

                int k;



                memcpy(temp_A, dpp->samplesA, sizeof(dpp->samplesA));

                memcpy(temp_B, dpp->samplesB, sizeof(dpp->samplesB));



                for (k = 0; k < MAX_TERM; k++) {

                    dpp->samplesA[k] = temp_A[m];

                    dpp->samplesB[k] = temp_B[m];

                    m = (m + 1) & (MAX_TERM - 1);

                }

            }

        }

    } else if (!s->num_passes) {

        if (s->flags & WV_JOINT_STEREO) {

            for (i = 0; i < nb_samples; i++)

                samples_r[i] += ((samples_l[i] -= samples_r[i]) >> 1);

        }



        for (i = 0; i < s->num_terms; i++) {

            struct Decorr *dpp = &s->decorr_passes[i];

            if (((s->flags & MAG_MASK) >> MAG_LSB) >= 16 || dpp->delta != 2)

                decorr_stereo_pass2(dpp, samples_l, samples_r, nb_samples);

            else

                decorr_stereo_pass_id2(dpp, samples_l, samples_r, nb_samples);

        }

    }



    bytestream2_put_byte(&pb, WP_ID_DATA | WP_IDF_LONG);

    init_put_bits(&s->pb, pb.buffer + 3, bytestream2_get_bytes_left_p(&pb));

    if (s->flags & WV_MONO_DATA) {

        for (i = 0; i < nb_samples; i++)

            wavpack_encode_sample(s, &s->w.c[0], s->samples[0][i]);

    } else {

        for (i = 0; i < nb_samples; i++) {

            wavpack_encode_sample(s, &s->w.c[0], s->samples[0][i]);

            wavpack_encode_sample(s, &s->w.c[1], s->samples[1][i]);

        }

    }

    encode_flush(s);

    flush_put_bits(&s->pb);

    data_size = put_bits_count(&s->pb) >> 3;

    bytestream2_put_le24(&pb, (data_size + 1) >> 1);

    bytestream2_skip_p(&pb, data_size);

    if (data_size & 1)

        bytestream2_put_byte(&pb, 0);



    if (got_extra) {

        bytestream2_put_byte(&pb, WP_ID_EXTRABITS | WP_IDF_LONG);

        init_put_bits(&s->pb, pb.buffer + 7, bytestream2_get_bytes_left_p(&pb));

        if (s->flags & WV_FLOAT_DATA)

            pack_float(s, s->orig_l, s->orig_r, nb_samples);

        else

            pack_int32(s, s->orig_l, s->orig_r, nb_samples);

        flush_put_bits(&s->pb);

        data_size = put_bits_count(&s->pb) >> 3;

        bytestream2_put_le24(&pb, (data_size + 5) >> 1);

        bytestream2_put_le32(&pb, s->crc_x);

        bytestream2_skip_p(&pb, data_size);

        if (data_size & 1)

            bytestream2_put_byte(&pb, 0);

    }



    block_size = bytestream2_tell_p(&pb);

    AV_WL32(out + 4, block_size - 8);



    return block_size;

}
