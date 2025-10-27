static int matroska_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    MatroskaDemuxContext *matroska = s->priv_data;

    int ret = 0;



    while (!ret && matroska_deliver_packet(matroska, pkt)) {

        if (matroska->done)

            return AVERROR_EOF;

        ret = matroska_parse_cluster(matroska);

    }



    return ret;

}
