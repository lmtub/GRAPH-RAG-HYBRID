matroska_parse_cluster (MatroskaDemuxContext *matroska)

{

    int res = 0;

    uint32_t id;

    uint64_t cluster_time = 0;

    uint8_t *data;

    int64_t pos;

    int size;



    av_log(matroska->ctx, AV_LOG_DEBUG,

           "parsing cluster at %"PRId64"\n", url_ftell(&matroska->ctx->pb));



    while (res == 0) {

        if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

            res = AVERROR_IO;

            break;

        } else if (matroska->level_up) {

            matroska->level_up--;

            break;

        }



        switch (id) {

            /* cluster timecode */

            case MATROSKA_ID_CLUSTERTIMECODE: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                cluster_time = num;

                break;

            }



                /* a group of blocks inside a cluster */

            case MATROSKA_ID_BLOCKGROUP:

                if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;

                res = matroska_parse_blockgroup(matroska, cluster_time);

                break;



            case MATROSKA_ID_SIMPLEBLOCK:

                pos = url_ftell(&matroska->ctx->pb);

                res = ebml_read_binary(matroska, &id, &data, &size);

                if (res == 0)

                    res = matroska_parse_block(matroska, data, size, pos,

                                               cluster_time, -1, NULL, NULL);

                break;



            default:

                av_log(matroska->ctx, AV_LOG_INFO,

                       "Unknown entry 0x%x in cluster data\n", id);

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



    return res;

}
