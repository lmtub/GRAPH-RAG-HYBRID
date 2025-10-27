AVParserState *ff_store_parser_state(AVFormatContext *s)

{

    int i;

    AVStream *st;

    AVParserStreamState *ss;

    AVParserState *state = av_malloc(sizeof(AVParserState));

    if (!state)

        return NULL;



    state->stream_states = av_malloc(sizeof(AVParserStreamState) * s->nb_streams);

    if (!state->stream_states) {

        av_free(state);

        return NULL;

    }



    state->fpos = avio_tell(s->pb);



    // copy context structures

    state->cur_st                           = s->cur_st;

    state->packet_buffer                    = s->packet_buffer;

    state->raw_packet_buffer                = s->raw_packet_buffer;

    state->raw_packet_buffer_remaining_size = s->raw_packet_buffer_remaining_size;



    s->cur_st                               = NULL;

    s->packet_buffer                        = NULL;

    s->raw_packet_buffer                    = NULL;

    s->raw_packet_buffer_remaining_size     = RAW_PACKET_BUFFER_SIZE;



    // copy stream structures

    state->nb_streams = s->nb_streams;

    for (i = 0; i < s->nb_streams; i++) {

        st = s->streams[i];

        ss = &state->stream_states[i];



        ss->parser        = st->parser;

        ss->last_IP_pts   = st->last_IP_pts;

        ss->cur_dts       = st->cur_dts;

        ss->reference_dts = st->reference_dts;

        ss->cur_ptr       = st->cur_ptr;

        ss->cur_len       = st->cur_len;

        ss->probe_packets = st->probe_packets;

        ss->cur_pkt       = st->cur_pkt;



        st->parser        = NULL;

        st->last_IP_pts   = AV_NOPTS_VALUE;

        st->cur_dts       = AV_NOPTS_VALUE;

        st->reference_dts = AV_NOPTS_VALUE;

        st->cur_ptr       = NULL;

        st->cur_len       = 0;

        st->probe_packets = MAX_PROBE_PACKETS;

        av_init_packet(&st->cur_pkt);

    }



    return state;

}
