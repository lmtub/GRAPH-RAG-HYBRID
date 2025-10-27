static int read_restart_header(MLPDecodeContext *m, GetBitContext *gbp,

                               const uint8_t *buf, unsigned int substr)

{

    SubStream *s = &m->substream[substr];

    unsigned int ch;

    int sync_word, tmp;

    uint8_t checksum;

    uint8_t lossless_check;

    int start_count = get_bits_count(gbp);

    const int max_matrix_channel = m->avctx->codec_id == CODEC_ID_MLP

                                 ? MAX_MATRIX_CHANNEL_MLP

                                 : MAX_MATRIX_CHANNEL_TRUEHD;



    sync_word = get_bits(gbp, 13);



    if (sync_word != 0x31ea >> 1) {

        av_log(m->avctx, AV_LOG_ERROR,

               "restart header sync incorrect (got 0x%04x)\n", sync_word);

        return AVERROR_INVALIDDATA;

    }



    s->noise_type = get_bits1(gbp);



    if (m->avctx->codec_id == CODEC_ID_MLP && s->noise_type) {

        av_log(m->avctx, AV_LOG_ERROR, "MLP must have 0x31ea sync word.\n");

        return AVERROR_INVALIDDATA;

    }



    skip_bits(gbp, 16); /* Output timestamp */



    s->min_channel        = get_bits(gbp, 4);

    s->max_channel        = get_bits(gbp, 4);

    s->max_matrix_channel = get_bits(gbp, 4);



    if (s->max_matrix_channel > max_matrix_channel) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Max matrix channel cannot be greater than %d.\n",

               max_matrix_channel);

        return AVERROR_INVALIDDATA;

    }



    if (s->max_channel != s->max_matrix_channel) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Max channel must be equal max matrix channel.\n");

        return AVERROR_INVALIDDATA;

    }



    /* This should happen for TrueHD streams with >6 channels and MLP's noise

     * type. It is not yet known if this is allowed. */

    if (s->max_channel > MAX_MATRIX_CHANNEL_MLP && !s->noise_type) {

        av_log_ask_for_sample(m->avctx,

               "Number of channels %d is larger than the maximum supported "

               "by the decoder.\n", s->max_channel + 2);

        return AVERROR_PATCHWELCOME;

    }



    if (s->min_channel > s->max_channel) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Substream min channel cannot be greater than max channel.\n");

        return AVERROR_INVALIDDATA;

    }



    if (m->avctx->request_channels > 0

        && s->max_channel + 1 >= m->avctx->request_channels

        && substr < m->max_decoded_substream) {

        av_log(m->avctx, AV_LOG_DEBUG,

               "Extracting %d channel downmix from substream %d. "

               "Further substreams will be skipped.\n",

               s->max_channel + 1, substr);

        m->max_decoded_substream = substr;

    }



    s->noise_shift   = get_bits(gbp,  4);

    s->noisegen_seed = get_bits(gbp, 23);



    skip_bits(gbp, 19);



    s->data_check_present = get_bits1(gbp);

    lossless_check = get_bits(gbp, 8);

    if (substr == m->max_decoded_substream

        && s->lossless_check_data != 0xffffffff) {

        tmp = xor_32_to_8(s->lossless_check_data);

        if (tmp != lossless_check)

            av_log(m->avctx, AV_LOG_WARNING,

                   "Lossless check failed - expected %02x, calculated %02x.\n",

                   lossless_check, tmp);

    }



    skip_bits(gbp, 16);



    memset(s->ch_assign, 0, sizeof(s->ch_assign));



    for (ch = 0; ch <= s->max_matrix_channel; ch++) {

        int ch_assign = get_bits(gbp, 6);

        if (ch_assign > s->max_matrix_channel) {

            av_log_ask_for_sample(m->avctx,

                   "Assignment of matrix channel %d to invalid output channel %d.\n",

                   ch, ch_assign);

            return AVERROR_PATCHWELCOME;

        }

        s->ch_assign[ch_assign] = ch;

    }



    if (m->avctx->codec_id == CODEC_ID_MLP && m->needs_reordering) {

        if (m->avctx->channel_layout == (AV_CH_LAYOUT_QUAD|AV_CH_LOW_FREQUENCY) ||

            m->avctx->channel_layout == AV_CH_LAYOUT_5POINT0_BACK) {

            int i = s->ch_assign[4];

            s->ch_assign[4] = s->ch_assign[3];

            s->ch_assign[3] = s->ch_assign[2];

            s->ch_assign[2] = i;

        } else if (m->avctx->channel_layout == AV_CH_LAYOUT_5POINT1_BACK) {

            FFSWAP(int, s->ch_assign[2], s->ch_assign[4]);

            FFSWAP(int, s->ch_assign[3], s->ch_assign[5]);

        }

    }

    if (m->avctx->codec_id == CODEC_ID_TRUEHD &&

        (m->avctx->channel_layout == AV_CH_LAYOUT_7POINT1 ||

        m->avctx->channel_layout == AV_CH_LAYOUT_7POINT1_WIDE)) {

        FFSWAP(int, s->ch_assign[4], s->ch_assign[6]);

        FFSWAP(int, s->ch_assign[5], s->ch_assign[7]);

    } else if (m->avctx->codec_id == CODEC_ID_TRUEHD &&

        (m->avctx->channel_layout == AV_CH_LAYOUT_6POINT1 ||

        m->avctx->channel_layout == (AV_CH_LAYOUT_6POINT1 | AV_CH_TOP_CENTER) ||

        m->avctx->channel_layout == (AV_CH_LAYOUT_6POINT1 | AV_CH_TOP_FRONT_CENTER))) {

        int i = s->ch_assign[6];

        s->ch_assign[6] = s->ch_assign[5];

        s->ch_assign[5] = s->ch_assign[4];

        s->ch_assign[4] = i;

    }



    checksum = ff_mlp_restart_checksum(buf, get_bits_count(gbp) - start_count);



    if (checksum != get_bits(gbp, 8))

        av_log(m->avctx, AV_LOG_ERROR, "restart header checksum error\n");



    /* Set default decoding parameters. */

    s->param_presence_flags   = 0xff;

    s->num_primitive_matrices = 0;

    s->blocksize              = 8;

    s->lossless_check_data    = 0;



    memset(s->output_shift   , 0, sizeof(s->output_shift   ));

    memset(s->quant_step_size, 0, sizeof(s->quant_step_size));



    for (ch = s->min_channel; ch <= s->max_channel; ch++) {

        ChannelParams *cp = &s->channel_params[ch];

        cp->filter_params[FIR].order = 0;

        cp->filter_params[IIR].order = 0;

        cp->filter_params[FIR].shift = 0;

        cp->filter_params[IIR].shift = 0;



        /* Default audio coding is 24-bit raw PCM. */

        cp->huff_offset      = 0;

        cp->sign_huff_offset = (-1) << 23;

        cp->codebook         = 0;

        cp->huff_lsbs        = 24;

    }



    if (substr == m->max_decoded_substream)

        m->avctx->channels = s->max_matrix_channel + 1;



    return 0;

}
