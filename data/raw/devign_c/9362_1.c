static int mov_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    MOVContext *mov = (MOVContext *) s->priv_data;

    ByteIOContext *pb = &s->pb;

    int i, err;

    MOV_atom_t atom = { 0, 0, 0 };



    mov->fc = s;

    mov->parse_table = mov_default_parse_table;



    if(!url_is_streamed(pb)) /* .mov and .mp4 aren't streamable anyway (only progressive download if moov is before mdat) */

        atom.size = url_fsize(pb);

    else

        atom.size = 0x7FFFFFFFFFFFFFFFLL;



    /* check MOV header */

    err = mov_read_default(mov, pb, atom);

    if (err<0 || (!mov->found_moov && !mov->found_mdat)) {

        av_log(s, AV_LOG_ERROR, "mov: header not found !!! (err:%d, moov:%d, mdat:%d) pos:%"PRId64"\n",

                err, mov->found_moov, mov->found_mdat, url_ftell(pb));

        return -1;

    }

    dprintf("on_parse_exit_offset=%d\n", (int) url_ftell(pb));



    /* some cleanup : make sure we are on the mdat atom */

    if(!url_is_streamed(pb) && (url_ftell(pb) != mov->mdat_offset))

        url_fseek(pb, mov->mdat_offset, SEEK_SET);



    mov->total_streams = s->nb_streams;



    for(i=0; i<mov->total_streams; i++) {

        MOVStreamContext *sc = mov->streams[i];



        if(!sc->time_rate)

            sc->time_rate=1;

        if(!sc->time_scale)

            sc->time_scale= mov->time_scale;

        av_set_pts_info(s->streams[i], 64, sc->time_rate, sc->time_scale);



        if(s->streams[i]->duration != AV_NOPTS_VALUE){

            assert(s->streams[i]->duration % sc->time_rate == 0);

            s->streams[i]->duration /= sc->time_rate;

        }

        sc->ffindex = i;

        mov_build_index(mov, s->streams[i]);

    }



    for(i=0; i<mov->total_streams; i++) {

        /* dont need those anymore */

        av_freep(&mov->streams[i]->chunk_offsets);

        av_freep(&mov->streams[i]->sample_to_chunk);

        av_freep(&mov->streams[i]->sample_sizes);

        av_freep(&mov->streams[i]->keyframes);

        av_freep(&mov->streams[i]->stts_data);

    }

    av_freep(&mov->mdat_list);

    return 0;

}
