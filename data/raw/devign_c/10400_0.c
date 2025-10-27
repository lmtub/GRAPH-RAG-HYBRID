void ff_free_parser_state(AVFormatContext *s, AVParserState *state)

{

    int i;

    AVParserStreamState *ss;



    if (!state)

        return;



    for (i = 0; i < state->nb_streams; i++) {

        ss = &state->stream_states[i];

        if (ss->parser)

            av_parser_close(ss->parser);

        av_free_packet(&ss->cur_pkt);

    }



    free_packet_list(state->packet_buffer);

    free_packet_list(state->raw_packet_buffer);



    av_free(state->stream_states);

    av_free(state);

}
