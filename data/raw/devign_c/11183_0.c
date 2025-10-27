static int mov_read_dec3(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    AVStream *st;

    enum AVAudioServiceType *ast;

    int eac3info, acmod, lfeon, bsmod;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];



    ast = (enum AVAudioServiceType*)ff_stream_new_side_data(st, AV_PKT_DATA_AUDIO_SERVICE_TYPE,

                                                            sizeof(*ast));

    if (!ast)

        return AVERROR(ENOMEM);



    /* No need to parse fields for additional independent substreams and its

     * associated dependent substreams since libavcodec's E-AC-3 decoder

     * does not support them yet. */

    avio_rb16(pb); /* data_rate and num_ind_sub */

    eac3info = avio_rb24(pb);

    bsmod = (eac3info >> 12) & 0x1f;

    acmod = (eac3info >>  9) & 0x7;

    lfeon = (eac3info >>  8) & 0x1;

    st->codec->channel_layout = avpriv_ac3_channel_layout_tab[acmod];

    if (lfeon)

        st->codec->channel_layout |= AV_CH_LOW_FREQUENCY;

    st->codec->channels = av_get_channel_layout_nb_channels(st->codec->channel_layout);

    *ast = bsmod;

    if (st->codec->channels > 1 && bsmod == 0x7)

        *ast = AV_AUDIO_SERVICE_TYPE_KARAOKE;



    st->codec->audio_service_type = *ast;



    return 0;

}
