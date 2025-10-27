matroska_read_packet (AVFormatContext *s,

                      AVPacket        *pkt)

{

    MatroskaDemuxContext *matroska = s->priv_data;

    int res = 0;

    uint32_t id;



    /* Read stream until we have a packet queued. */

    while (matroska_deliver_packet(matroska, pkt)) {



        /* Have we already reached the end? */

        if (matroska->done)

            return AVERROR_IO;



        while (res == 0) {

            if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

                return AVERROR_IO;

            } else if (matroska->level_up) {

                matroska->level_up--;

                break;

            }



            switch (id) {

                case MATROSKA_ID_CLUSTER:

                    if ((res = ebml_read_master(matroska, &id)) < 0)

                        break;

                    if ((res = matroska_parse_cluster(matroska)) == 0)

                        res = 1; /* Parsed one cluster, let's get out. */

                    break;



                default:

                case EBML_ID_VOID:

                    res = ebml_read_skip(matroska);

                    break;

            }



            if (matroska->level_up) {

                matroska->level_up--;

                break;

            }

        }



        if (res == -1)

            matroska->done = 1;

    }



    return 0;

}
