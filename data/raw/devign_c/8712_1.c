static av_cold int iss_read_header(AVFormatContext *s)

{

    IssDemuxContext *iss = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st;

    char token[MAX_TOKEN_SIZE];

    int stereo, rate_divisor;



    get_token(pb, token, sizeof(token)); //"IMA_ADPCM_Sound"

    get_token(pb, token, sizeof(token)); //packet size

    sscanf(token, "%d", &iss->packet_size);

    get_token(pb, token, sizeof(token)); //File ID

    get_token(pb, token, sizeof(token)); //out size

    get_token(pb, token, sizeof(token)); //stereo

    sscanf(token, "%d", &stereo);

    get_token(pb, token, sizeof(token)); //Unknown1

    get_token(pb, token, sizeof(token)); //RateDivisor

    sscanf(token, "%d", &rate_divisor);

    get_token(pb, token, sizeof(token)); //Unknown2

    get_token(pb, token, sizeof(token)); //Version ID

    get_token(pb, token, sizeof(token)); //Size



    if (iss->packet_size <= 0) {

        av_log(s, AV_LOG_ERROR, "packet_size %d is invalid\n", iss->packet_size);

        return AVERROR_INVALIDDATA;

    }



    iss->sample_start_pos = avio_tell(pb);



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id = AV_CODEC_ID_ADPCM_IMA_ISS;

    if (stereo) {

        st->codec->channels       = 2;

        st->codec->channel_layout = AV_CH_LAYOUT_STEREO;

    } else {

        st->codec->channels       = 1;

        st->codec->channel_layout = AV_CH_LAYOUT_MONO;

    }

    st->codec->sample_rate = 44100;

    if(rate_divisor > 0)

         st->codec->sample_rate /= rate_divisor;

    st->codec->bits_per_coded_sample = 4;

    st->codec->bit_rate = st->codec->channels * st->codec->sample_rate

                                      * st->codec->bits_per_coded_sample;

    st->codec->block_align = iss->packet_size;

    avpriv_set_pts_info(st, 32, 1, st->codec->sample_rate);



    return 0;

}
