static int read_restart_header(MLPDecodeContext *m, BitstreamContext *bc,

                               const uint8_t *buf, unsigned int substr)

{

    SubStream *s = &m->substream[substr];

    unsigned int ch;

    int sync_word, tmp;

    uint8_t checksum;

    uint8_t lossless_check;

    int start_count = bitstream_tell(bc);

    int min_channel, max_channel, max_matrix_channel;

    const int std_max_matrix_channel = m->avctx->codec_id == AV_CODEC_ID_MLP

                                     ? MAX_MATRIX_CHANNEL_MLP

                                     : MAX_MATRIX_CHANNEL_TRUEHD;



    sync_word = bitstream_read(bc, 13);



    if (sync_word != 0x31ea >> 1) {

        av_log(m->avctx, AV_LOG_ERROR,

               "restart header sync incorrect (got 0x%04x)\n", sync_word);

        return AVERROR_INVALIDDATA;

    }



    s->noise_type = bitstream_read_bit(bc);



    if (m->avctx->codec_id == AV_CODEC_ID_MLP && s->noise_type) {

        av_log(m->avctx, AV_LOG_ERROR, "MLP must have 0x31ea sync word.\n");

        return AVERROR_INVALIDDATA;

    }



    bitstream_skip(bc, 16); /* Output timestamp */



    min_channel        = bitstream_read(bc, 4);

    max_channel        = bitstream_read(bc, 4);

    max_matrix_channel = bitstream_read(bc, 4);



    if (max_matrix_channel > std_max_matrix_channel) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Max matrix channel cannot be greater than %d.\n",

               max_matrix_channel);

        return AVERROR_INVALIDDATA;

    }



    if (max_channel != max_matrix_channel) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Max channel must be equal max matrix channel.\n");

        return AVERROR_INVALIDDATA;

    }



    /* This should happen for TrueHD streams with >6 channels and MLP's noise

     * type. It is not yet known if this is allowed. */

    if (s->max_channel > MAX_MATRIX_CHANNEL_MLP && !s->noise_type) {

        avpriv_request_sample(m->avctx,

                              "%d channels (more than the "

                              "maximum supported by the decoder)",

                              s->max_channel + 2);

        return AVERROR_PATCHWELCOME;

    }



    if (min_channel > max_channel) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Substream min channel cannot be greater than max channel.\n");

        return AVERROR_INVALIDDATA;

    }



    s->min_channel        = min_channel;

    s->max_channel        = max_channel;

    s->max_matrix_channel = max_matrix_channel;



    if (m->avctx->request_channel_layout && (s->mask & m->avctx->request_channel_layout) ==

        m->avctx->request_channel_layout && m->max_decoded_substream > substr) {

        av_log(m->avctx, AV_LOG_DEBUG,

               "Extracting %d-channel downmix (0x%"PRIx64") from substream %d. "

               "Further substreams will be skipped.\n",

               s->max_channel + 1, s->mask, substr);

        m->max_decoded_substream = substr;

    }



    s->noise_shift   = bitstream_read(bc,  4);

    s->noisegen_seed = bitstream_read(bc, 23);



    bitstream_skip(bc, 19);



    s->data_check_present = bitstream_read_bit(bc);

    lossless_check = bitstream_read(bc, 8);

    if (substr == m->max_decoded_substream

        && s->lossless_check_data != 0xffffffff) {

        tmp = xor_32_to_8(s->lossless_check_data);

        if (tmp != lossless_check)

            av_log(m->avctx, AV_LOG_WARNING,

                   "Lossless check failed - expected %02x, calculated %02x.\n",

                   lossless_check, tmp);

    }



    bitstream_skip(bc, 16);



    memset(s->ch_assign, 0, sizeof(s->ch_assign));



    for (ch = 0; ch <= s->max_matrix_channel; ch++) {

        int ch_assign = bitstream_read(bc, 6);

        if (m->avctx->codec_id == AV_CODEC_ID_TRUEHD) {

            uint64_t channel = thd_channel_layout_extract_channel(s->mask,

                                                                  ch_assign);

            ch_assign = av_get_channel_layout_channel_index(s->mask,

                                                            channel);

        }

        if (ch_assign < 0 || ch_assign > s->max_matrix_channel) {

            avpriv_request_sample(m->avctx,

                                  "Assignment of matrix channel %d to invalid output channel %d",

                                  ch, ch_assign);

            return AVERROR_PATCHWELCOME;

        }

        s->ch_assign[ch_assign] = ch;

    }



    checksum = ff_mlp_restart_checksum(buf, bitstream_tell(bc) - start_count);



    if (checksum != bitstream_read(bc, 8))

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

        cp->sign_huff_offset = -(1 << 23);

        cp->codebook         = 0;

        cp->huff_lsbs        = 24;

    }



    if (substr == m->max_decoded_substream) {

        m->avctx->channels       = s->max_matrix_channel + 1;

        m->avctx->channel_layout = s->mask;

        m->dsp.mlp_pack_output = m->dsp.mlp_select_pack_output(s->ch_assign,

                                                               s->output_shift,

                                                               s->max_matrix_channel,

                                                               m->avctx->sample_fmt == AV_SAMPLE_FMT_S32);

    }



    return 0;

}
