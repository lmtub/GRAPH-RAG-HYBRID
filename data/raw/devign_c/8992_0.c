static void asf_build_simple_index(AVFormatContext *s, int stream_index)

{

    ff_asf_guid g;

    ASFContext *asf = s->priv_data;

    int64_t current_pos= avio_tell(s->pb);

    int i;



    avio_seek(s->pb, asf->data_object_offset + asf->data_object_size, SEEK_SET);

    ff_get_guid(s->pb, &g);



    /* the data object can be followed by other top-level objects,

       skip them until the simple index object is reached */

    while (ff_guidcmp(&g, &index_guid)) {

        int64_t gsize= avio_rl64(s->pb);

        if (gsize < 24 || url_feof(s->pb)) {

            avio_seek(s->pb, current_pos, SEEK_SET);

            return;

        }

        avio_skip(s->pb, gsize-24);

        ff_get_guid(s->pb, &g);

    }



    {

        int64_t itime, last_pos=-1;

        int pct, ict;

        int64_t av_unused gsize= avio_rl64(s->pb);

        ff_get_guid(s->pb, &g);

        itime=avio_rl64(s->pb);

        pct=avio_rl32(s->pb);

        ict=avio_rl32(s->pb);

        av_log(s, AV_LOG_DEBUG, "itime:0x%"PRIx64", pct:%d, ict:%d\n",itime,pct,ict);



        for (i=0;i<ict;i++){

            int pktnum=avio_rl32(s->pb);

            int pktct =avio_rl16(s->pb);

            int64_t pos      = s->data_offset + s->packet_size*(int64_t)pktnum;

            int64_t index_pts= FFMAX(av_rescale(itime, i, 10000) - asf->hdr.preroll, 0);



            if(pos != last_pos){

            av_log(s, AV_LOG_DEBUG, "pktnum:%d, pktct:%d  pts: %"PRId64"\n", pktnum, pktct, index_pts);

            av_add_index_entry(s->streams[stream_index], pos, index_pts, s->packet_size, 0, AVINDEX_KEYFRAME);

            last_pos=pos;

            }

        }

        asf->index_read= 1;

    }

    avio_seek(s->pb, current_pos, SEEK_SET);

}
