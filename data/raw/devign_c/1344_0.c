int ff_flac_decode_frame_header(AVCodecContext *avctx, GetBitContext *gb,

                                FLACFrameInfo *fi)

{

    int bs_code, sr_code, bps_code;



    /* frame sync code */

    skip_bits(gb, 16);



    /* block size and sample rate codes */

    bs_code = get_bits(gb, 4);

    sr_code = get_bits(gb, 4);



    /* channels and decorrelation */

    fi->ch_mode = get_bits(gb, 4);

    if (fi->ch_mode < FLAC_MAX_CHANNELS) {

        fi->channels = fi->ch_mode + 1;

        fi->ch_mode = FLAC_CHMODE_INDEPENDENT;

    } else if (fi->ch_mode <= FLAC_CHMODE_MID_SIDE) {

        fi->channels = 2;

    } else {

        av_log(avctx, AV_LOG_ERROR, "invalid channel mode: %d\n", fi->ch_mode);

        return -1;

    }



    /* bits per sample */

    bps_code = get_bits(gb, 3);

    if (bps_code == 3 || bps_code == 7) {

        av_log(avctx, AV_LOG_ERROR, "invalid sample size code (%d)\n",

               bps_code);

        return -1;

    }

    fi->bps = sample_size_table[bps_code];



    /* reserved bit */

    if (get_bits1(gb)) {

        av_log(avctx, AV_LOG_ERROR, "broken stream, invalid padding\n");

        return -1;

    }



    /* sample or frame count */

    if (get_utf8(gb) < 0) {

        av_log(avctx, AV_LOG_ERROR, "utf8 fscked\n");

        return -1;

    }



    /* blocksize */

    if (bs_code == 0) {

        av_log(avctx, AV_LOG_ERROR, "reserved blocksize code: 0\n");

        return -1;

    } else if (bs_code == 6) {

        fi->blocksize = get_bits(gb, 8) + 1;

    } else if (bs_code == 7) {

        fi->blocksize = get_bits(gb, 16) + 1;

    } else {

        fi->blocksize = ff_flac_blocksize_table[bs_code];

    }



    /* sample rate */

    if (sr_code < 12) {

        fi->samplerate = ff_flac_sample_rate_table[sr_code];

    } else if (sr_code == 12) {

        fi->samplerate = get_bits(gb, 8) * 1000;

    } else if (sr_code == 13) {

        fi->samplerate = get_bits(gb, 16);

    } else if (sr_code == 14) {

        fi->samplerate = get_bits(gb, 16) * 10;

    } else {

        av_log(avctx, AV_LOG_ERROR, "illegal sample rate code %d\n",

               sr_code);

        return -1;

    }



    /* header CRC-8 check */

    skip_bits(gb, 8);

    if (av_crc(av_crc_get_table(AV_CRC_8_ATM), 0, gb->buffer,

               get_bits_count(gb)/8)) {

        av_log(avctx, AV_LOG_ERROR, "header crc mismatch\n");

        return -1;

    }



    return 0;

}
