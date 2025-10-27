static int wavpack_decode_block(AVCodecContext *avctx, int block_no,

                                AVFrame *frame, const uint8_t *buf, int buf_size)

{

    WavpackContext *wc = avctx->priv_data;

    ThreadFrame tframe = { .f = frame };

    WavpackFrameContext *s;

    GetByteContext gb;

    void *samples_l = NULL, *samples_r = NULL;

    int ret;

    int got_terms   = 0, got_weights = 0, got_samples = 0,

        got_entropy = 0, got_bs      = 0, got_float   = 0, got_hybrid = 0;

    int i, j, id, size, ssize, weights, t;

    int bpp, chan = 0, chmask = 0, orig_bpp, sample_rate = 0;

    int multiblock;



    if (block_no >= wc->fdec_num && wv_alloc_frame_context(wc) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error creating frame decode context\n");

        return AVERROR_INVALIDDATA;

    }



    s = wc->fdec[block_no];

    if (!s) {

        av_log(avctx, AV_LOG_ERROR, "Context for block %d is not present\n",

               block_no);

        return AVERROR_INVALIDDATA;

    }



    memset(s->decorr, 0, MAX_TERMS * sizeof(Decorr));

    memset(s->ch, 0, sizeof(s->ch));

    s->extra_bits     = 0;

    s->and            = s->or = s->shift = 0;

    s->got_extra_bits = 0;



    bytestream2_init(&gb, buf, buf_size);



    s->samples = bytestream2_get_le32(&gb);

    if (s->samples != wc->samples) {

        av_log(avctx, AV_LOG_ERROR, "Mismatching number of samples in "

               "a sequence: %d and %d\n", wc->samples, s->samples);

        return AVERROR_INVALIDDATA;

    }

    s->frame_flags = bytestream2_get_le32(&gb);

    bpp            = av_get_bytes_per_sample(avctx->sample_fmt);

    orig_bpp       = ((s->frame_flags & 0x03) + 1) << 3;

    multiblock     = (s->frame_flags & WV_SINGLE_BLOCK) != WV_SINGLE_BLOCK;



    s->stereo         = !(s->frame_flags & WV_MONO);

    s->stereo_in      =  (s->frame_flags & WV_FALSE_STEREO) ? 0 : s->stereo;

    s->joint          =   s->frame_flags & WV_JOINT_STEREO;

    s->hybrid         =   s->frame_flags & WV_HYBRID_MODE;

    s->hybrid_bitrate =   s->frame_flags & WV_HYBRID_BITRATE;

    s->post_shift     = bpp * 8 - orig_bpp + ((s->frame_flags >> 13) & 0x1f);

    s->hybrid_maxclip =  ((1LL << (orig_bpp - 1)) - 1);

    s->hybrid_minclip = ((-1LL << (orig_bpp - 1)));

    s->CRC            = bytestream2_get_le32(&gb);



    // parse metadata blocks

    while (bytestream2_get_bytes_left(&gb)) {

        id   = bytestream2_get_byte(&gb);

        size = bytestream2_get_byte(&gb);

        if (id & WP_IDF_LONG) {

            size |= (bytestream2_get_byte(&gb)) << 8;

            size |= (bytestream2_get_byte(&gb)) << 16;

        }

        size <<= 1; // size is specified in words

        ssize  = size;

        if (id & WP_IDF_ODD)

            size--;

        if (size < 0) {

            av_log(avctx, AV_LOG_ERROR,

                   "Got incorrect block %02X with size %i\n", id, size);

            break;

        }

        if (bytestream2_get_bytes_left(&gb) < ssize) {

            av_log(avctx, AV_LOG_ERROR,

                   "Block size %i is out of bounds\n", size);

            break;

        }

        switch (id & WP_IDF_MASK) {

        case WP_ID_DECTERMS:

            if (size > MAX_TERMS) {

                av_log(avctx, AV_LOG_ERROR, "Too many decorrelation terms\n");

                s->terms = 0;

                bytestream2_skip(&gb, ssize);

                continue;

            }

            s->terms = size;

            for (i = 0; i < s->terms; i++) {

                uint8_t val = bytestream2_get_byte(&gb);

                s->decorr[s->terms - i - 1].value = (val & 0x1F) - 5;

                s->decorr[s->terms - i - 1].delta =  val >> 5;

            }

            got_terms = 1;

            break;

        case WP_ID_DECWEIGHTS:

            if (!got_terms) {

                av_log(avctx, AV_LOG_ERROR, "No decorrelation terms met\n");

                continue;

            }

            weights = size >> s->stereo_in;

            if (weights > MAX_TERMS || weights > s->terms) {

                av_log(avctx, AV_LOG_ERROR, "Too many decorrelation weights\n");

                bytestream2_skip(&gb, ssize);

                continue;

            }

            for (i = 0; i < weights; i++) {

                t = (int8_t)bytestream2_get_byte(&gb);

                s->decorr[s->terms - i - 1].weightA = t << 3;

                if (s->decorr[s->terms - i - 1].weightA > 0)

                    s->decorr[s->terms - i - 1].weightA +=

                        (s->decorr[s->terms - i - 1].weightA + 64) >> 7;

                if (s->stereo_in) {

                    t = (int8_t)bytestream2_get_byte(&gb);

                    s->decorr[s->terms - i - 1].weightB = t << 3;

                    if (s->decorr[s->terms - i - 1].weightB > 0)

                        s->decorr[s->terms - i - 1].weightB +=

                            (s->decorr[s->terms - i - 1].weightB + 64) >> 7;

                }

            }

            got_weights = 1;

            break;

        case WP_ID_DECSAMPLES:

            if (!got_terms) {

                av_log(avctx, AV_LOG_ERROR, "No decorrelation terms met\n");

                continue;

            }

            t = 0;

            for (i = s->terms - 1; (i >= 0) && (t < size); i--) {

                if (s->decorr[i].value > 8) {

                    s->decorr[i].samplesA[0] =

                        wp_exp2(bytestream2_get_le16(&gb));

                    s->decorr[i].samplesA[1] =

                        wp_exp2(bytestream2_get_le16(&gb));



                    if (s->stereo_in) {

                        s->decorr[i].samplesB[0] =

                            wp_exp2(bytestream2_get_le16(&gb));

                        s->decorr[i].samplesB[1] =

                            wp_exp2(bytestream2_get_le16(&gb));

                        t                       += 4;

                    }

                    t += 4;

                } else if (s->decorr[i].value < 0) {

                    s->decorr[i].samplesA[0] =

                        wp_exp2(bytestream2_get_le16(&gb));

                    s->decorr[i].samplesB[0] =

                        wp_exp2(bytestream2_get_le16(&gb));

                    t                       += 4;

                } else {

                    for (j = 0; j < s->decorr[i].value; j++) {

                        s->decorr[i].samplesA[j] =

                            wp_exp2(bytestream2_get_le16(&gb));

                        if (s->stereo_in) {

                            s->decorr[i].samplesB[j] =

                                wp_exp2(bytestream2_get_le16(&gb));

                        }

                    }

                    t += s->decorr[i].value * 2 * (s->stereo_in + 1);

                }

            }

            got_samples = 1;

            break;

        case WP_ID_ENTROPY:

            if (size != 6 * (s->stereo_in + 1)) {

                av_log(avctx, AV_LOG_ERROR,

                       "Entropy vars size should be %i, got %i.\n",

                       6 * (s->stereo_in + 1), size);

                bytestream2_skip(&gb, ssize);

                continue;

            }

            for (j = 0; j <= s->stereo_in; j++)

                for (i = 0; i < 3; i++) {

                    s->ch[j].median[i] = wp_exp2(bytestream2_get_le16(&gb));

                }

            got_entropy = 1;

            break;

        case WP_ID_HYBRID:

            if (s->hybrid_bitrate) {

                for (i = 0; i <= s->stereo_in; i++) {

                    s->ch[i].slow_level = wp_exp2(bytestream2_get_le16(&gb));

                    size               -= 2;

                }

            }

            for (i = 0; i < (s->stereo_in + 1); i++) {

                s->ch[i].bitrate_acc = bytestream2_get_le16(&gb) << 16;

                size                -= 2;

            }

            if (size > 0) {

                for (i = 0; i < (s->stereo_in + 1); i++) {

                    s->ch[i].bitrate_delta =

                        wp_exp2((int16_t)bytestream2_get_le16(&gb));

                }

            } else {

                for (i = 0; i < (s->stereo_in + 1); i++)

                    s->ch[i].bitrate_delta = 0;

            }

            got_hybrid = 1;

            break;

        case WP_ID_INT32INFO: {

            uint8_t val[4];

            if (size != 4) {

                av_log(avctx, AV_LOG_ERROR,

                       "Invalid INT32INFO, size = %i\n",

                       size);

                bytestream2_skip(&gb, ssize - 4);

                continue;

            }

            bytestream2_get_buffer(&gb, val, 4);

            if (val[0] > 32) {

                av_log(avctx, AV_LOG_ERROR,

                       "Invalid INT32INFO, extra_bits = %d (> 32)\n", val[0]);

                continue;

            } else if (val[0]) {

                s->extra_bits = val[0];

            } else if (val[1]) {

                s->shift = val[1];

            } else if (val[2]) {

                s->and   = s->or = 1;

                s->shift = val[2];

            } else if (val[3]) {

                s->and   = 1;

                s->shift = val[3];

            }

            /* original WavPack decoder forces 32-bit lossy sound to be treated

             * as 24-bit one in order to have proper clipping */

            if (s->hybrid && bpp == 4 && s->post_shift < 8 && s->shift > 8) {

                s->post_shift      += 8;

                s->shift           -= 8;

                s->hybrid_maxclip >>= 8;

                s->hybrid_minclip >>= 8;

            }

            break;

        }

        case WP_ID_FLOATINFO:

            if (size != 4) {

                av_log(avctx, AV_LOG_ERROR,

                       "Invalid FLOATINFO, size = %i\n", size);

                bytestream2_skip(&gb, ssize);

                continue;

            }

            s->float_flag    = bytestream2_get_byte(&gb);

            s->float_shift   = bytestream2_get_byte(&gb);

            s->float_max_exp = bytestream2_get_byte(&gb);

            got_float        = 1;

            bytestream2_skip(&gb, 1);

            break;

        case WP_ID_DATA:

            s->sc.offset = bytestream2_tell(&gb);

            s->sc.size   = size * 8;

            if ((ret = init_get_bits8(&s->gb, gb.buffer, size)) < 0)

                return ret;

            s->data_size = size * 8;

            bytestream2_skip(&gb, size);

            got_bs       = 1;

            break;

        case WP_ID_EXTRABITS:

            if (size <= 4) {

                av_log(avctx, AV_LOG_ERROR, "Invalid EXTRABITS, size = %i\n",

                       size);

                bytestream2_skip(&gb, size);

                continue;

            }

            s->extra_sc.offset = bytestream2_tell(&gb);

            s->extra_sc.size   = size * 8;

            if ((ret = init_get_bits8(&s->gb_extra_bits, gb.buffer, size)) < 0)

                return ret;

            s->crc_extra_bits  = get_bits_long(&s->gb_extra_bits, 32);

            bytestream2_skip(&gb, size);

            s->got_extra_bits  = 1;

            break;

        case WP_ID_CHANINFO:

            if (size <= 1) {

                av_log(avctx, AV_LOG_ERROR,

                       "Insufficient channel information\n");

                return AVERROR_INVALIDDATA;

            }

            chan = bytestream2_get_byte(&gb);

            switch (size - 2) {

            case 0:

                chmask = bytestream2_get_byte(&gb);

                break;

            case 1:

                chmask = bytestream2_get_le16(&gb);

                break;

            case 2:

                chmask = bytestream2_get_le24(&gb);

                break;

            case 3:

                chmask = bytestream2_get_le32(&gb);

                break;

            case 5:

                size = bytestream2_get_byte(&gb);

                if (avctx->channels != size)

                    av_log(avctx, AV_LOG_WARNING, "%i channels signalled"

                           " instead of %i.\n", size, avctx->channels);

                chan  |= (bytestream2_get_byte(&gb) & 0xF) << 8;

                chmask = bytestream2_get_le16(&gb);

                break;

            default:

                av_log(avctx, AV_LOG_ERROR, "Invalid channel info size %d\n",

                       size);

                chan   = avctx->channels;

                chmask = avctx->channel_layout;

            }

            break;

        case WP_ID_SAMPLE_RATE:

            if (size != 3) {

                av_log(avctx, AV_LOG_ERROR, "Invalid custom sample rate.\n");

                return AVERROR_INVALIDDATA;

            }

            sample_rate = bytestream2_get_le24(&gb);

            break;

        default:

            bytestream2_skip(&gb, size);

        }

        if (id & WP_IDF_ODD)

            bytestream2_skip(&gb, 1);

    }



    if (!got_terms) {

        av_log(avctx, AV_LOG_ERROR, "No block with decorrelation terms\n");

        return AVERROR_INVALIDDATA;

    }

    if (!got_weights) {

        av_log(avctx, AV_LOG_ERROR, "No block with decorrelation weights\n");

        return AVERROR_INVALIDDATA;

    }

    if (!got_samples) {

        av_log(avctx, AV_LOG_ERROR, "No block with decorrelation samples\n");

        return AVERROR_INVALIDDATA;

    }

    if (!got_entropy) {

        av_log(avctx, AV_LOG_ERROR, "No block with entropy info\n");

        return AVERROR_INVALIDDATA;

    }

    if (s->hybrid && !got_hybrid) {

        av_log(avctx, AV_LOG_ERROR, "Hybrid config not found\n");

        return AVERROR_INVALIDDATA;

    }

    if (!got_bs) {

        av_log(avctx, AV_LOG_ERROR, "Packed samples not found\n");

        return AVERROR_INVALIDDATA;

    }

    if (!got_float && avctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {

        av_log(avctx, AV_LOG_ERROR, "Float information not found\n");

        return AVERROR_INVALIDDATA;

    }

    if (s->got_extra_bits && avctx->sample_fmt != AV_SAMPLE_FMT_FLTP) {

        const int size   = get_bits_left(&s->gb_extra_bits);

        const int wanted = s->samples * s->extra_bits << s->stereo_in;

        if (size < wanted) {

            av_log(avctx, AV_LOG_ERROR, "Too small EXTRABITS\n");

            s->got_extra_bits = 0;

        }

    }



    if (!wc->ch_offset) {

        int sr = (s->frame_flags >> 23) & 0xf;

        if (sr == 0xf) {

            if (!sample_rate) {

                av_log(avctx, AV_LOG_ERROR, "Custom sample rate missing.\n");

                return AVERROR_INVALIDDATA;

            }

            avctx->sample_rate = sample_rate;

        } else

            avctx->sample_rate = wv_rates[sr];



        if (multiblock) {

            if (chan)

                avctx->channels = chan;

            if (chmask)

                avctx->channel_layout = chmask;

        } else {

            avctx->channels       = s->stereo ? 2 : 1;

            avctx->channel_layout = s->stereo ? AV_CH_LAYOUT_STEREO :

                                                AV_CH_LAYOUT_MONO;

        }



        /* get output buffer */

        frame->nb_samples = s->samples + 1;

        if ((ret = ff_thread_get_buffer(avctx, &tframe, 0)) < 0)

            return ret;

        frame->nb_samples = s->samples;

    }



    if (wc->ch_offset + s->stereo >= avctx->channels) {

        av_log(avctx, AV_LOG_WARNING, "Too many channels coded in a packet.\n");

        return (avctx->err_recognition & AV_EF_EXPLODE) ? AVERROR_INVALIDDATA : 0;

    }



    samples_l = frame->extended_data[wc->ch_offset];

    if (s->stereo)

        samples_r = frame->extended_data[wc->ch_offset + 1];



    wc->ch_offset += 1 + s->stereo;



    if (s->stereo_in) {

        ret = wv_unpack_stereo(s, &s->gb, samples_l, samples_r, avctx->sample_fmt);

        if (ret < 0)

            return ret;

    } else {

        ret = wv_unpack_mono(s, &s->gb, samples_l, avctx->sample_fmt);

        if (ret < 0)

            return ret;



        if (s->stereo)

            memcpy(samples_r, samples_l, bpp * s->samples);

    }



    return 0;

}
