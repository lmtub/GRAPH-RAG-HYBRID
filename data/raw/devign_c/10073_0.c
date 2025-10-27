static int rtp_write_header(AVFormatContext *s1)

{

    RTPDemuxContext *s = s1->priv_data;

    int payload_type, max_packet_size, n;

    AVStream *st;



    if (s1->nb_streams != 1)

        return -1;

    st = s1->streams[0];



    payload_type = rtp_get_payload_type(st->codec);

    if (payload_type < 0)

        payload_type = RTP_PT_PRIVATE; /* private payload type */

    s->payload_type = payload_type;



    s->base_timestamp = random();

    s->timestamp = s->base_timestamp;

    s->ssrc = random();

    s->first_packet = 1;



    max_packet_size = url_fget_max_packet_size(&s1->pb);

    if (max_packet_size <= 12)

        return AVERROR_IO;

    s->max_payload_size = max_packet_size - 12;



    switch(st->codec->codec_id) {

    case CODEC_ID_MP2:

    case CODEC_ID_MP3:

        s->buf_ptr = s->buf + 4;

        s->cur_timestamp = 0;

        break;

    case CODEC_ID_MPEG1VIDEO:

        s->cur_timestamp = 0;

        break;

    case CODEC_ID_MPEG2TS:

        n = s->max_payload_size / TS_PACKET_SIZE;

        if (n < 1)

            n = 1;

        s->max_payload_size = n * TS_PACKET_SIZE;

        s->buf_ptr = s->buf;

        break;

    default:

        s->buf_ptr = s->buf;

        break;

    }



    return 0;

}
