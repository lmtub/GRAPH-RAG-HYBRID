matroska_parse_blockgroup (MatroskaDemuxContext *matroska,

                           uint64_t              cluster_time)

{

    int res = 0;

    uint32_t id;

    AVPacket *pkt = NULL;

    int is_keyframe = PKT_FLAG_KEY, last_num_packets = matroska->num_packets;

    uint64_t duration = AV_NOPTS_VALUE;

    int track = -1;

    uint8_t *data;

    int size = 0;

    int64_t pos = 0;



    av_log(matroska->ctx, AV_LOG_DEBUG, "parsing blockgroup...\n");



    while (res == 0) {

        if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

            res = AVERROR_IO;

            break;

        } else if (matroska->level_up) {

            matroska->level_up--;

            break;

        }



        switch (id) {

            /* one block inside the group. Note, block parsing is one

             * of the harder things, so this code is a bit complicated.

             * See http://www.matroska.org/ for documentation. */

            case MATROSKA_ID_BLOCK: {

                pos = url_ftell(&matroska->ctx->pb);

                res = ebml_read_binary(matroska, &id, &data, &size);

                break;

            }



            case MATROSKA_ID_BLOCKDURATION: {

                if ((res = ebml_read_uint(matroska, &id, &duration)) < 0)

                    break;

                break;

            }



            case MATROSKA_ID_BLOCKREFERENCE:

                /* We've found a reference, so not even the first frame in

                 * the lace is a key frame. */

                is_keyframe = 0;

                if (last_num_packets != matroska->num_packets)

                    matroska->packets[last_num_packets]->flags = 0;

                res = ebml_read_skip(matroska);

                break;



            default:

                av_log(matroska->ctx, AV_LOG_INFO,

                       "Unknown entry 0x%x in blockgroup data\n", id);

                /* fall-through */



            case EBML_ID_VOID:

                res = ebml_read_skip(matroska);

                break;

        }



        if (matroska->level_up) {

            matroska->level_up--;

            break;

        }

    }



    if (res)

        return res;



    if (size > 0)

        res = matroska_parse_block(matroska, data, size, pos, cluster_time,

                                   is_keyframe, &track, &pkt);



    if (pkt)

    {

        if (duration != AV_NOPTS_VALUE)

            pkt->duration = duration;

        else if (track >= 0 && track < matroska->num_tracks)

            pkt->duration = matroska->tracks[track]->default_duration / matroska->time_scale;

    }



    return res;

}
