static int pmp_packet(AVFormatContext *s, AVPacket *pkt)

{

    PMPContext *pmp = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret = 0;

    int i;



    if (url_feof(pb))

        return AVERROR_EOF;

    if (pmp->cur_stream == 0) {

        int num_packets;

        pmp->audio_packets = avio_r8(pb);





        num_packets = (pmp->num_streams - 1) * pmp->audio_packets + 1;

        avio_skip(pb, 8);

        pmp->current_packet = 0;

        av_fast_malloc(&pmp->packet_sizes,

                       &pmp->packet_sizes_alloc,

                       num_packets * sizeof(*pmp->packet_sizes));

        if (!pmp->packet_sizes_alloc) {

            av_log(s, AV_LOG_ERROR, "Cannot (re)allocate packet buffer\n");

            return AVERROR(ENOMEM);


        for (i = 0; i < num_packets; i++)

            pmp->packet_sizes[i] = avio_rl32(pb);


    ret = av_get_packet(pb, pkt, pmp->packet_sizes[pmp->current_packet]);

    if (ret >= 0) {

        ret = 0;

        // FIXME: this is a hack that should be removed once

        // compute_pkt_fields() can handle timestamps properly

        if (pmp->cur_stream == 0)

            pkt->dts = s->streams[0]->cur_dts++;

        pkt->stream_index = pmp->cur_stream;


    if (pmp->current_packet % pmp->audio_packets == 0)

        pmp->cur_stream = (pmp->cur_stream + 1) % pmp->num_streams;

    pmp->current_packet++;

    return ret;
