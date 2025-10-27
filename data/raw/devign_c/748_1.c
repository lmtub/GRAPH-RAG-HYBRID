static int read_major_sync(MLPDecodeContext *m, GetBitContext *gb)

{

    MLPHeaderInfo mh;

    int substr, ret;



    if ((ret = ff_mlp_read_major_sync(m->avctx, &mh, gb)) != 0)

        return ret;



    if (mh.group1_bits == 0) {

        av_log(m->avctx, AV_LOG_ERROR, "invalid/unknown bits per sample\n");

        return AVERROR_INVALIDDATA;


    if (mh.group2_bits > mh.group1_bits) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Channel group 2 cannot have more bits per sample than group 1.\n");

        return AVERROR_INVALIDDATA;




    if (mh.group2_samplerate && mh.group2_samplerate != mh.group1_samplerate) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Channel groups with differing sample rates are not currently supported.\n");

        return AVERROR_INVALIDDATA;




    if (mh.group1_samplerate == 0) {

        av_log(m->avctx, AV_LOG_ERROR, "invalid/unknown sampling rate\n");

        return AVERROR_INVALIDDATA;


    if (mh.group1_samplerate > MAX_SAMPLERATE) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Sampling rate %d is greater than the supported maximum (%d).\n",

               mh.group1_samplerate, MAX_SAMPLERATE);

        return AVERROR_INVALIDDATA;


    if (mh.access_unit_size > MAX_BLOCKSIZE) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Block size %d is greater than the supported maximum (%d).\n",

               mh.access_unit_size, MAX_BLOCKSIZE);

        return AVERROR_INVALIDDATA;


    if (mh.access_unit_size_pow2 > MAX_BLOCKSIZE_POW2) {

        av_log(m->avctx, AV_LOG_ERROR,

               "Block size pow2 %d is greater than the supported maximum (%d).\n",

               mh.access_unit_size_pow2, MAX_BLOCKSIZE_POW2);

        return AVERROR_INVALIDDATA;




    if (mh.num_substreams == 0)

        return AVERROR_INVALIDDATA;

    if (m->avctx->codec_id == AV_CODEC_ID_MLP && mh.num_substreams > 2) {

        av_log(m->avctx, AV_LOG_ERROR, "MLP only supports up to 2 substreams.\n");

        return AVERROR_INVALIDDATA;


    if (mh.num_substreams > MAX_SUBSTREAMS) {


                              "%d substreams (more than the "

                              "maximum supported by the decoder)",

                              mh.num_substreams);





    m->access_unit_size      = mh.access_unit_size;

    m->access_unit_size_pow2 = mh.access_unit_size_pow2;



    m->num_substreams        = mh.num_substreams;

    m->max_decoded_substream = m->num_substreams - 1;



    m->avctx->sample_rate    = mh.group1_samplerate;

    m->avctx->frame_size     = mh.access_unit_size;



    m->avctx->bits_per_raw_sample = mh.group1_bits;

    if (mh.group1_bits > 16)

        m->avctx->sample_fmt = AV_SAMPLE_FMT_S32;

    else

        m->avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    m->params_valid = 1;

    for (substr = 0; substr < MAX_SUBSTREAMS; substr++)

        m->substream[substr].restart_seen = 0;



    /* Set the layout for each substream. When there's more than one, the first

     * substream is Stereo. Subsequent substreams' layouts are indicated in the

     * major sync. */

    if (m->avctx->codec_id == AV_CODEC_ID_MLP) {







        if ((substr = (mh.num_substreams > 1)))

            m->substream[0].ch_layout = AV_CH_LAYOUT_STEREO;

        m->substream[substr].ch_layout = mh.channel_layout_mlp;

    } else {







        if ((substr = (mh.num_substreams > 1)))

            m->substream[0].ch_layout = AV_CH_LAYOUT_STEREO;

        if (mh.num_substreams > 2)

            if (mh.channel_layout_thd_stream2)

                m->substream[2].ch_layout = mh.channel_layout_thd_stream2;

            else

                m->substream[2].ch_layout = mh.channel_layout_thd_stream1;

        m->substream[substr].ch_layout = mh.channel_layout_thd_stream1;



        if (m->avctx->channels<=2 && m->substream[substr].ch_layout == AV_CH_LAYOUT_MONO && m->max_decoded_substream == 1) {

            av_log(m->avctx, AV_LOG_DEBUG, "Mono stream with 2 substreams, ignoring 2nd\n");

            m->max_decoded_substream = 0;

            if (m->avctx->channels==2)

                m->avctx->channel_layout = AV_CH_LAYOUT_STEREO;





    m->needs_reordering = mh.channel_arrangement >= 18 && mh.channel_arrangement <= 20;



    return 0;
