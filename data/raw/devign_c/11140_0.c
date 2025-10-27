DVDemuxContext* dv_init_demux(AVFormatContext *s)

{

    DVDemuxContext *c;



    c = av_mallocz(sizeof(DVDemuxContext));

    if (!c)

        return NULL;



    c->vst = av_new_stream(s, 0);

    c->ast[0] = av_new_stream(s, 0);

    if (!c->vst || !c->ast[0])

        goto fail;

    av_set_pts_info(c->vst, 64, 1, 30000);

    av_set_pts_info(c->ast[0], 64, 1, 30000);



    c->fctx = s;

    c->ast[1] = NULL;

    c->ach = 0;

    c->frames = 0;

    c->abytes = 0;

    c->audio_pkt[0].size = 0;

    c->audio_pkt[1].size = 0;

    

    c->vst->codec.codec_type = CODEC_TYPE_VIDEO;

    c->vst->codec.codec_id = CODEC_ID_DVVIDEO;

    c->vst->codec.bit_rate = 25000000;

    

    c->ast[0]->codec.codec_type = CODEC_TYPE_AUDIO;

    c->ast[0]->codec.codec_id = CODEC_ID_PCM_S16LE;

   

    s->ctx_flags |= AVFMTCTX_NOHEADER; 

    

    return c;

    

fail:

    if (c->vst)

        av_free(c->vst);

    if (c->ast[0])

        av_free(c->ast[0]);

    av_free(c);

    return NULL;

}
