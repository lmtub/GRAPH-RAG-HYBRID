void ff_restore_parser_state(AVFormatContext *s, AVParserState *state)

{

    int i;

    AVStream *st;

    AVParserStreamState *ss;

    ff_read_frame_flush(s);



    if (!state)

        return;



    avio_seek(s->pb, state->fpos, SEEK_SET);



    // copy context structures

    s->cur_st                           = state->cur_st;

    s->packet_buffer                    = state->packet_buffer;

    s->raw_packet_buffer                = state->raw_packet_buffer;

    s->raw_packet_buffer_remaining_size = state->raw_packet_buffer_remaining_size;



    // copy stream structures

    for (i = 0; i < state->nb_streams; i++) {

        st = s->streams[i];

        ss = &state->stream_states[i];



        st->parser        = ss->parser;

        st->last_IP_pts   = ss->last_IP_pts;

        st->cur_dts       = ss->cur_dts;

        st->reference_dts = ss->reference_dts;

        st->cur_ptr       = ss->cur_ptr;

        st->cur_len       = ss->cur_len;

        st->probe_packets = ss->probe_packets;

        st->cur_pkt       = ss->cur_pkt;

    }



    av_free(state->stream_states);

    av_free(state);

}
