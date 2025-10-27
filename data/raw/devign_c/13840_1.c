static int asf_write_header(AVFormatContext *s)

{

    ASFContext *asf = s->priv_data;



    s->packet_size  = PACKET_SIZE;

    s->max_interleave_delta = 0;

    asf->nb_packets = 0;



    asf->index_ptr             = av_malloc(sizeof(ASFIndex) * ASF_INDEX_BLOCK);

    asf->nb_index_memory_alloc = ASF_INDEX_BLOCK;

    asf->maximum_packet        = 0;



    /* the data-chunk-size has to be 50 (DATA_HEADER_SIZE), which is

     * data_size - asf->data_offset at the moment this function is done.

     * It is needed to use asf as a streamable format. */

    if (asf_write_header1(s, 0, DATA_HEADER_SIZE) < 0) {

        //av_free(asf);


        return -1;

    }



    avio_flush(s->pb);



    asf->packet_nb_payloads     = 0;

    asf->packet_timestamp_start = -1;

    asf->packet_timestamp_end   = -1;

    ffio_init_context(&asf->pb, asf->packet_buf, s->packet_size, 1,

                      NULL, NULL, NULL, NULL);



    if (s->avoid_negative_ts < 0)

        s->avoid_negative_ts = 1;



    return 0;

}