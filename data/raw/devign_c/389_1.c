static int smacker_read_header(AVFormatContext *s, AVFormatParameters *ap)
{
    ByteIOContext *pb = &s->pb;
    SmackerContext *smk = (SmackerContext *)s->priv_data;
    AVStream *st, *ast[7];
    int i, ret;
    int tbase;
    /* read and check header */
    smk->magic = get_le32(pb);
    if (smk->magic != MKTAG('S', 'M', 'K', '2') && smk->magic != MKTAG('S', 'M', 'K', '4'))
    smk->width = get_le32(pb);
    smk->height = get_le32(pb);
    smk->frames = get_le32(pb);
    smk->pts_inc = (int32_t)get_le32(pb);
    smk->flags = get_le32(pb);
    for(i = 0; i < 7; i++)
        smk->audio[i] = get_le32(pb);
    smk->treesize = get_le32(pb);
    smk->mmap_size = get_le32(pb);
    smk->mclr_size = get_le32(pb);
    smk->full_size = get_le32(pb);
    smk->type_size = get_le32(pb);
    for(i = 0; i < 7; i++)
        smk->rates[i] = get_le32(pb);
    smk->pad = get_le32(pb);
    /* setup data */
    if(smk->frames > 0xFFFFFF) {
        av_log(s, AV_LOG_ERROR, "Too many frames: %i\n", smk->frames);
    smk->frm_size = av_malloc(smk->frames * 4);
    smk->frm_flags = av_malloc(smk->frames);
    smk->is_ver4 = (smk->magic != MKTAG('S', 'M', 'K', '2'));
    /* read frame info */
    for(i = 0; i < smk->frames; i++) {
        smk->frm_size[i] = get_le32(pb);
    for(i = 0; i < smk->frames; i++) {
        smk->frm_flags[i] = get_byte(pb);
    /* init video codec */
    st = av_new_stream(s, 0);
    if (!st)
    smk->videoindex = st->index;
    st->codec->width = smk->width;
    st->codec->height = smk->height;
    st->codec->pix_fmt = PIX_FMT_PAL8;
    st->codec->codec_type = CODEC_TYPE_VIDEO;
    st->codec->codec_id = CODEC_ID_SMACKVIDEO;
    st->codec->codec_tag = smk->is_ver4;
    /* Smacker uses 100000 as internal timebase */
    if(smk->pts_inc < 0)
        smk->pts_inc = -smk->pts_inc;
    else
        smk->pts_inc *= 100;
    tbase = 100000;
    av_reduce(&tbase, &smk->pts_inc, tbase, smk->pts_inc, (1UL<<31)-1);
    av_set_pts_info(st, 33, smk->pts_inc, tbase);
    /* handle possible audio streams */
    for(i = 0; i < 7; i++) {
        smk->indexes[i] = -1;
        if((smk->rates[i] & 0xFFFFFF) && !(smk->rates[i] & SMK_AUD_BINKAUD)){
            ast[i] = av_new_stream(s, 0);
            smk->indexes[i] = ast[i]->index;
            av_set_pts_info(ast[i], 33, smk->pts_inc, tbase);
            ast[i]->codec->codec_type = CODEC_TYPE_AUDIO;
            ast[i]->codec->codec_id = (smk->rates[i] & SMK_AUD_PACKED) ? CODEC_ID_SMACKAUDIO : CODEC_ID_PCM_U8;
            ast[i]->codec->codec_tag = 0;
            ast[i]->codec->channels = (smk->rates[i] & SMK_AUD_STEREO) ? 2 : 1;
            ast[i]->codec->sample_rate = smk->rates[i] & 0xFFFFFF;
            ast[i]->codec->bits_per_sample = (smk->rates[i] & SMK_AUD_16BITS) ? 16 : 8;
            if(ast[i]->codec->bits_per_sample == 16 && ast[i]->codec->codec_id == CODEC_ID_PCM_U8)
                ast[i]->codec->codec_id = CODEC_ID_PCM_S16LE;
    /* load trees to extradata, they will be unpacked by decoder */
    st->codec->extradata = av_malloc(smk->treesize + 16);
    st->codec->extradata_size = smk->treesize + 16;
    if(!st->codec->extradata){
        av_log(s, AV_LOG_ERROR, "Cannot allocate %i bytes of extradata\n", smk->treesize + 16);
        av_free(smk->frm_size);
        av_free(smk->frm_flags);
    ret = get_buffer(pb, st->codec->extradata + 16, st->codec->extradata_size - 16);
    if(ret != st->codec->extradata_size - 16){
        av_free(smk->frm_size);
        av_free(smk->frm_flags);
        return AVERROR_IO;
    ((int32_t*)st->codec->extradata)[0] = le2me_32(smk->mmap_size);
    ((int32_t*)st->codec->extradata)[1] = le2me_32(smk->mclr_size);
    ((int32_t*)st->codec->extradata)[2] = le2me_32(smk->full_size);
    ((int32_t*)st->codec->extradata)[3] = le2me_32(smk->type_size);
    smk->curstream = -1;
    smk->nextpos = url_ftell(pb);
    return 0;