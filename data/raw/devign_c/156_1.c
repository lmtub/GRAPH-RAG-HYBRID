int av_get_audio_frame_duration(AVCodecContext *avctx, int frame_bytes)

{

    int id, sr, ch, ba, tag, bps;



    id  = avctx->codec_id;

    sr  = avctx->sample_rate;

    ch  = avctx->channels;

    ba  = avctx->block_align;

    tag = avctx->codec_tag;

    bps = av_get_exact_bits_per_sample(avctx->codec_id);



    /* codecs with an exact constant bits per sample */

    if (bps > 0 && ch > 0 && frame_bytes > 0)

        return (frame_bytes * 8) / (bps * ch);

    bps = avctx->bits_per_coded_sample;



    /* codecs with a fixed packet duration */

    switch (id) {

    case CODEC_ID_ADPCM_ADX:    return   32;

    case CODEC_ID_ADPCM_IMA_QT: return   64;

    case CODEC_ID_ADPCM_EA_XAS: return  128;

    case CODEC_ID_AMR_NB:

    case CODEC_ID_GSM:

    case CODEC_ID_QCELP:

    case CODEC_ID_RA_144:

    case CODEC_ID_RA_288:       return  160;

    case CODEC_ID_IMC:          return  256;

    case CODEC_ID_AMR_WB:

    case CODEC_ID_GSM_MS:       return  320;

    case CODEC_ID_MP1:          return  384;

    case CODEC_ID_ATRAC1:       return  512;

    case CODEC_ID_ATRAC3:       return 1024;

    case CODEC_ID_MP2:

    case CODEC_ID_MUSEPACK7:    return 1152;

    case CODEC_ID_AC3:          return 1536;

    }



    if (sr > 0) {

        /* calc from sample rate */

        if (id == CODEC_ID_TTA)

            return 256 * sr / 245;



        if (ch > 0) {

            /* calc from sample rate and channels */

            if (id == CODEC_ID_BINKAUDIO_DCT)

                return (480 << (sr / 22050)) / ch;

        }

    }



    if (ba > 0) {

        /* calc from block_align */

        if (id == CODEC_ID_SIPR) {

            switch (ba) {

            case 20: return 160;

            case 19: return 144;

            case 29: return 288;

            case 37: return 480;

            }

        }

    }



    if (frame_bytes > 0) {

        /* calc from frame_bytes only */

        if (id == CODEC_ID_TRUESPEECH)

            return 240 * (frame_bytes / 32);

        if (id == CODEC_ID_NELLYMOSER)

            return 256 * (frame_bytes / 64);



        if (bps > 0) {

            /* calc from frame_bytes and bits_per_coded_sample */

            if (id == CODEC_ID_ADPCM_G726)

                return frame_bytes * 8 / bps;

        }



        if (ch > 0) {

            /* calc from frame_bytes and channels */

            switch (id) {

            case CODEC_ID_ADPCM_4XM:

            case CODEC_ID_ADPCM_IMA_ISS:

                return (frame_bytes - 4 * ch) * 2 / ch;

            case CODEC_ID_ADPCM_IMA_SMJPEG:

                return (frame_bytes - 4) * 2 / ch;

            case CODEC_ID_ADPCM_IMA_AMV:

                return (frame_bytes - 8) * 2 / ch;

            case CODEC_ID_ADPCM_XA:

                return (frame_bytes / 128) * 224 / ch;

            case CODEC_ID_INTERPLAY_DPCM:

                return (frame_bytes - 6 - ch) / ch;

            case CODEC_ID_ROQ_DPCM:

                return (frame_bytes - 8) / ch;

            case CODEC_ID_XAN_DPCM:

                return (frame_bytes - 2 * ch) / ch;

            case CODEC_ID_MACE3:

                return 3 * frame_bytes / ch;

            case CODEC_ID_MACE6:

                return 6 * frame_bytes / ch;

            case CODEC_ID_PCM_LXF:

                return 2 * (frame_bytes / (5 * ch));

            }



            if (tag) {

                /* calc from frame_bytes, channels, and codec_tag */

                if (id == CODEC_ID_SOL_DPCM) {

                    if (tag == 3)

                        return frame_bytes / ch;

                    else

                        return frame_bytes * 2 / ch;

                }

            }



            if (ba > 0) {

                /* calc from frame_bytes, channels, and block_align */

                int blocks = frame_bytes / ba;

                switch (avctx->codec_id) {

                case CODEC_ID_ADPCM_IMA_WAV:

                    return blocks * (1 + (ba - 4 * ch) / (4 * ch) * 8);

                case CODEC_ID_ADPCM_IMA_DK3:

                    return blocks * (((ba - 16) * 2 / 3 * 4) / ch);

                case CODEC_ID_ADPCM_IMA_DK4:

                    return blocks * (1 + (ba - 4 * ch) * 2 / ch);

                case CODEC_ID_ADPCM_MS:

                    return blocks * (2 + (ba - 7 * ch) * 2 / ch);

                }

            }



            if (bps > 0) {

                /* calc from frame_bytes, channels, and bits_per_coded_sample */

                switch (avctx->codec_id) {

                case CODEC_ID_PCM_DVD:

                    return 2 * (frame_bytes / ((bps * 2 / 8) * ch));

                case CODEC_ID_PCM_BLURAY:

                    return frame_bytes / ((FFALIGN(ch, 2) * bps) / 8);

                case CODEC_ID_S302M:

                    return 2 * (frame_bytes / ((bps + 4) / 4)) / ch;

                }

            }

        }

    }



    return 0;

}
