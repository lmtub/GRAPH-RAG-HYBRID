ff_rdt_parse_open(AVFormatContext *ic, int first_stream_of_set_idx,

                  void *priv_data, RTPDynamicProtocolHandler *handler)

{

    RDTDemuxContext *s = av_mallocz(sizeof(RDTDemuxContext));

    if (!s)

        return NULL;



    s->ic = ic;

    s->streams = &ic->streams[first_stream_of_set_idx];

    do {

        s->n_streams++;

    } while (first_stream_of_set_idx + s->n_streams < ic->nb_streams &&

             s->streams[s->n_streams]->priv_data == s->streams[0]->priv_data);

    s->prev_set_id    = -1;

    s->prev_stream_id = -1;

    s->prev_timestamp = -1;

    s->parse_packet = handler->parse_packet;

    s->dynamic_protocol_context = priv_data;



    return s;

}
