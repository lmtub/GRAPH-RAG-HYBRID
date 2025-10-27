static int smacker_read_header(AVFormatContext *s)
{
    AVIOContext *pb = s->pb;
    SmackerContext *smk = s->priv_data;
    AVStream *st, *ast[7];
    int i, ret;
    int tbase;
    /* read and check header */
    smk->magic = avio_rl32(pb);
    if (smk->magic != MKTAG('S', 'M', 'K', '2') && smk->magic != MKTAG('S', 'M', 'K', '4'))
    smk->width = avio_rl32(pb);
    smk->height = avio_rl32(pb);
    smk->frames = avio_rl32(pb);
    smk->pts_inc = (int32_t)avio_rl32(pb);
    smk->flags = avio_rl32(pb);
    if(smk->flags & SMACKER_FLAG_RING_FRAME)
        smk->frames++;
    for(i = 0; i < 7; i++)
        smk->audio[i] = avio_rl32(pb);
    smk->treesize = avio_rl32(pb);
    if(smk->treesize >= UINT_MAX/4){ // smk->treesize + 16 must not overflow (this check is probably redundant)
        av_log(s, AV_LOG_ERROR, "treesize too large\n");
//FIXME remove extradata "rebuilding"
    smk->mmap_size = avio_rl32(pb);
    smk->mclr_size = avio_rl32(pb);
    smk->full_size = avio_rl32(pb);
    smk->type_size = avio_rl32(pb);
    for(i = 0; i < 7; i++) {
        smk->rates[i]  = avio_rl24(pb);
        smk->aflags[i] = avio_r8(pb);
    smk->pad = avio_rl32(pb);
    /* setup data */
    if(smk->frames > 0xFFFFFF) {
        av_log(s, AV_LOG_ERROR, "Too many frames: %"PRIu32"\n", smk->frames);
    smk->frm_size = av_malloc_array(smk->frames, sizeof(*smk->frm_size));
    smk->frm_flags = av_malloc(smk->frames);
    if (!smk->frm_size || !smk->frm_flags) {
        av_freep(&smk->frm_size);
        av_freep(&smk->frm_flags);
        return AVERROR(ENOMEM);
    smk->is_ver4 = (smk->magic != MKTAG('S', 'M', 'K', '2'));
    /* read frame info */
    for(i = 0; i < smk->frames; i++) {
        smk->frm_size[i] = avio_rl32(pb);
    for(i = 0; i < smk->frames; i++) {
        smk->frm_flags[i] = avio_r8(pb);
    /* init video codec */
    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
    smk->videoindex = st->index;
    st->codec->width = smk->width;
    st->codec->height = smk->height;
    st->codec->pix_fmt = AV_PIX_FMT_PAL8;
    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codec->codec_id = AV_CODEC_ID_SMACKVIDEO;
    st->codec->codec_tag = smk->magic;
    /* Smacker uses 100000 as internal timebase */
    if(smk->pts_inc < 0)
        smk->pts_inc = -smk->pts_inc;
    else
        smk->pts_inc *= 100;
    tbase = 100000;
    av_reduce(&tbase, &smk->pts_inc, tbase, smk->pts_inc, (1UL<<31)-1);
    avpriv_set_pts_info(st, 33, smk->pts_inc, tbase);
    st->duration = smk->frames;
    /* handle possible audio streams */
    for(i = 0; i < 7; i++) {
        smk->indexes[i] = -1;
        if (smk->rates[i]) {
            ast[i] = avformat_new_stream(s, NULL);
            if (!ast[i])
                return AVERROR(ENOMEM);
            smk->indexes[i] = ast[i]->index;
            ast[i]->codec->codec_type = AVMEDIA_TYPE_AUDIO;
            if (smk->aflags[i] & SMK_AUD_BINKAUD) {
                ast[i]->codec->codec_id = AV_CODEC_ID_BINKAUDIO_RDFT;
            } else if (smk->aflags[i] & SMK_AUD_USEDCT) {
                ast[i]->codec->codec_id = AV_CODEC_ID_BINKAUDIO_DCT;
            } else if (smk->aflags[i] & SMK_AUD_PACKED){
                ast[i]->codec->codec_id = AV_CODEC_ID_SMACKAUDIO;
                ast[i]->codec->codec_tag = MKTAG('S', 'M', 'K', 'A');
            } else {
                ast[i]->codec->codec_id = AV_CODEC_ID_PCM_U8;
            if (smk->aflags[i] & SMK_AUD_STEREO) {
                ast[i]->codec->channels       = 2;
                ast[i]->codec->channel_layout = AV_CH_LAYOUT_STEREO;
            } else {
                ast[i]->codec->channels       = 1;
                ast[i]->codec->channel_layout = AV_CH_LAYOUT_MONO;
            ast[i]->codec->sample_rate = smk->rates[i];
            ast[i]->codec->bits_per_coded_sample = (smk->aflags[i] & SMK_AUD_16BITS) ? 16 : 8;
            if(ast[i]->codec->bits_per_coded_sample == 16 && ast[i]->codec->codec_id == AV_CODEC_ID_PCM_U8)
                ast[i]->codec->codec_id = AV_CODEC_ID_PCM_S16LE;
            avpriv_set_pts_info(ast[i], 64, 1, ast[i]->codec->sample_rate
                    * ast[i]->codec->channels * ast[i]->codec->bits_per_coded_sample / 8);
    /* load trees to extradata, they will be unpacked by decoder */
    if(ff_alloc_extradata(st->codec, smk->treesize + 16)){
        av_log(s, AV_LOG_ERROR,
               "Cannot allocate %"PRIu32" bytes of extradata\n",
               smk->treesize + 16);
        av_freep(&smk->frm_size);
        av_freep(&smk->frm_flags);
        return AVERROR(ENOMEM);
    ret = avio_read(pb, st->codec->extradata + 16, st->codec->extradata_size - 16);
    if(ret != st->codec->extradata_size - 16){
        av_freep(&smk->frm_size);
        av_freep(&smk->frm_flags);
        return AVERROR(EIO);
    ((int32_t*)st->codec->extradata)[0] = av_le2ne32(smk->mmap_size);
    ((int32_t*)st->codec->extradata)[1] = av_le2ne32(smk->mclr_size);
    ((int32_t*)st->codec->extradata)[2] = av_le2ne32(smk->full_size);
    ((int32_t*)st->codec->extradata)[3] = av_le2ne32(smk->type_size);
    smk->curstream = -1;
    smk->nextpos = avio_tell(pb);
    return 0;