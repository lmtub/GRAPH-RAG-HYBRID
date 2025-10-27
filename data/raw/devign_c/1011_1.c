static int matroska_parse_cluster_incremental(MatroskaDemuxContext *matroska)

{

    EbmlList *blocks_list;

    MatroskaBlock *blocks;

    int i, res;

    res = ebml_parse(matroska,

                     matroska_cluster_incremental_parsing,

                     &matroska->current_cluster);

    if (res == 1) {

        /* New Cluster */

        if (matroska->current_cluster_pos)

            ebml_level_end(matroska);

        ebml_free(matroska_cluster, &matroska->current_cluster);

        memset(&matroska->current_cluster, 0, sizeof(MatroskaCluster));

        matroska->current_cluster_num_blocks = 0;

        matroska->current_cluster_pos = avio_tell(matroska->ctx->pb);

        matroska->prev_pkt = NULL;

        /* sizeof the ID which was already read */

        if (matroska->current_id)

            matroska->current_cluster_pos -= 4;

        res = ebml_parse(matroska,

                         matroska_clusters_incremental,

                         &matroska->current_cluster);

        /* Try parsing the block again. */

        if (res == 1)

            res = ebml_parse(matroska,

                             matroska_cluster_incremental_parsing,

                             &matroska->current_cluster);

    }



    if (!res &&

        matroska->current_cluster_num_blocks <

            matroska->current_cluster.blocks.nb_elem) {

        blocks_list = &matroska->current_cluster.blocks;

        blocks = blocks_list->elem;



        matroska->current_cluster_num_blocks = blocks_list->nb_elem;

        i = blocks_list->nb_elem - 1;

        if (blocks[i].bin.size > 0 && blocks[i].bin.data) {

            int is_keyframe = blocks[i].non_simple ? !blocks[i].reference : -1;

            uint8_t* additional = blocks[i].additional.size > 0 ?

                                    blocks[i].additional.data : NULL;

            if (!blocks[i].non_simple)

                blocks[i].duration = 0;

            res = matroska_parse_block(matroska,

                                       blocks[i].bin.data, blocks[i].bin.size,

                                       blocks[i].bin.pos,

                                       matroska->current_cluster.timecode,

                                       blocks[i].duration, is_keyframe,

                                       additional, blocks[i].additional_id,

                                       blocks[i].additional.size,

                                       matroska->current_cluster_pos);

        }

    }



    if (res < 0)  matroska->done = 1;

    return res;

}
