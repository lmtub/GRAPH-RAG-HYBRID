int attribute_align_arg avcodec_open2(AVCodecContext *avctx, AVCodec *codec, AVDictionary **options)
{
    int ret = 0;
    AVDictionary *tmp = NULL;
    if (avctx->extradata_size < 0 || avctx->extradata_size >= FF_MAX_EXTRADATA_SIZE)
        return AVERROR(EINVAL);
    if (options)
        av_dict_copy(&tmp, *options, 0);
    /* If there is a user-supplied mutex locking routine, call it. */
    if (ff_lockmgr_cb) {
        if ((*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_OBTAIN))
            return -1;
    entangled_thread_counter++;
    if(entangled_thread_counter != 1){
        av_log(avctx, AV_LOG_ERROR, "insufficient thread locking around avcodec_open/close()\n");
        goto end;
    if(avctx->codec || !codec) {
        ret = AVERROR(EINVAL);
        goto end;
    avctx->internal = av_mallocz(sizeof(AVCodecInternal));
    if (!avctx->internal) {
        ret = AVERROR(ENOMEM);
        goto end;
    if (codec->priv_data_size > 0) {
      if(!avctx->priv_data){
        avctx->priv_data = av_mallocz(codec->priv_data_size);
        if (!avctx->priv_data) {
            ret = AVERROR(ENOMEM);
            goto end;
        if (codec->priv_class) {
            *(AVClass**)avctx->priv_data= codec->priv_class;
            av_opt_set_defaults(avctx->priv_data);
      if (codec->priv_class && (ret = av_opt_set_dict(avctx->priv_data, &tmp)) < 0)
    } else {
        avctx->priv_data = NULL;
    if ((ret = av_opt_set_dict(avctx, &tmp)) < 0)
    //We only call avcodec_set_dimensions() for non h264 codecs so as not to overwrite previously setup dimensions
    if(!( avctx->coded_width && avctx->coded_height && avctx->width && avctx->height && avctx->codec_id == CODEC_ID_H264)){
    if(avctx->coded_width && avctx->coded_height)
        avcodec_set_dimensions(avctx, avctx->coded_width, avctx->coded_height);
    else if(avctx->width && avctx->height)
        avcodec_set_dimensions(avctx, avctx->width, avctx->height);
    if ((avctx->coded_width || avctx->coded_height || avctx->width || avctx->height)
        && (  av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx) < 0
           || av_image_check_size(avctx->width,       avctx->height,       0, avctx) < 0)) {
        av_log(avctx, AV_LOG_WARNING, "ignoring invalid width/height values\n");
        avcodec_set_dimensions(avctx, 0, 0);
    /* if the decoder init function was already called previously,
       free the already allocated subtitle_header before overwriting it */
    if (codec->decode)
        av_freep(&avctx->subtitle_header);
#define SANE_NB_CHANNELS 128U
    if (avctx->channels > SANE_NB_CHANNELS) {
        ret = AVERROR(EINVAL);
    avctx->codec = codec;
    if ((avctx->codec_type == AVMEDIA_TYPE_UNKNOWN || avctx->codec_type == codec->type) &&
        avctx->codec_id == CODEC_ID_NONE) {
        avctx->codec_type = codec->type;
        avctx->codec_id   = codec->id;
    if (avctx->codec_id != codec->id || (avctx->codec_type != codec->type
                           && avctx->codec_type != AVMEDIA_TYPE_ATTACHMENT)) {
        av_log(avctx, AV_LOG_ERROR, "codec type or id mismatches\n");
        ret = AVERROR(EINVAL);
    avctx->frame_number = 0;
#if FF_API_ER
    av_log(avctx, AV_LOG_DEBUG, "err{or,}_recognition separate: %d; %X\n",
           avctx->error_recognition, avctx->err_recognition);
    switch(avctx->error_recognition){
        case FF_ER_EXPLODE        : avctx->err_recognition |= AV_EF_EXPLODE | AV_EF_COMPLIANT | AV_EF_CAREFUL;
            break;
        case FF_ER_VERY_AGGRESSIVE:
        case FF_ER_AGGRESSIVE     : avctx->err_recognition |= AV_EF_AGGRESSIVE;
        case FF_ER_COMPLIANT      : avctx->err_recognition |= AV_EF_COMPLIANT;
        case FF_ER_CAREFUL        : avctx->err_recognition |= AV_EF_CAREFUL;
    av_log(avctx, AV_LOG_DEBUG, "err{or,}_recognition combined: %d; %X\n",
           avctx->error_recognition, avctx->err_recognition);
#endif
    if (!HAVE_THREADS)
        av_log(avctx, AV_LOG_WARNING, "Warning: not compiled with thread support, using thread emulation\n");
    if (HAVE_THREADS && !avctx->thread_opaque) {
        ret = ff_thread_init(avctx);
        if (ret < 0) {
    if (!HAVE_THREADS && !(codec->capabilities & CODEC_CAP_AUTO_THREADS))
        avctx->thread_count = 1;
    if (avctx->codec->max_lowres < avctx->lowres || avctx->lowres < 0) {
        av_log(avctx, AV_LOG_ERROR, "The maximum value for lowres supported by the decoder is %d\n",
               avctx->codec->max_lowres);
        ret = AVERROR(EINVAL);
    if (avctx->codec->encode) {
        int i;
        if (avctx->codec->sample_fmts) {
            for (i = 0; avctx->codec->sample_fmts[i] != AV_SAMPLE_FMT_NONE; i++)
                if (avctx->sample_fmt == avctx->codec->sample_fmts[i])
                    break;
            if (avctx->codec->sample_fmts[i] == AV_SAMPLE_FMT_NONE) {
                av_log(avctx, AV_LOG_ERROR, "Specified sample_fmt is not supported.\n");
                ret = AVERROR(EINVAL);
        if (avctx->codec->supported_samplerates) {
            for (i = 0; avctx->codec->supported_samplerates[i] != 0; i++)
                if (avctx->sample_rate == avctx->codec->supported_samplerates[i])
                    break;
            if (avctx->codec->supported_samplerates[i] == 0) {
                av_log(avctx, AV_LOG_ERROR, "Specified sample_rate is not supported\n");
                ret = AVERROR(EINVAL);
        if (avctx->codec->channel_layouts) {
            if (!avctx->channel_layout) {
                av_log(avctx, AV_LOG_WARNING, "channel_layout not specified\n");
            } else {
                for (i = 0; avctx->codec->channel_layouts[i] != 0; i++)
                    if (avctx->channel_layout == avctx->codec->channel_layouts[i])
                        break;
                if (avctx->codec->channel_layouts[i] == 0) {
                    av_log(avctx, AV_LOG_ERROR, "Specified channel_layout is not supported\n");
                    ret = AVERROR(EINVAL);
        if (avctx->channel_layout && avctx->channels) {
            if (av_get_channel_layout_nb_channels(avctx->channel_layout) != avctx->channels) {
                av_log(avctx, AV_LOG_ERROR, "channel layout does not match number of channels\n");
                ret = AVERROR(EINVAL);
        } else if (avctx->channel_layout) {
            avctx->channels = av_get_channel_layout_nb_channels(avctx->channel_layout);
    avctx->pts_correction_num_faulty_pts =
    avctx->pts_correction_num_faulty_dts = 0;
    avctx->pts_correction_last_pts =
    avctx->pts_correction_last_dts = INT64_MIN;
    if(avctx->codec->init && !(avctx->active_thread_type&FF_THREAD_FRAME)){
        ret = avctx->codec->init(avctx);
        if (ret < 0) {
    ret=0;
end:
    entangled_thread_counter--;
    /* Release any user-supplied mutex. */
    if (ff_lockmgr_cb) {
        (*ff_lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE);
    if (options) {
        av_dict_free(options);
        *options = tmp;
    return ret;
free_and_end:
    av_dict_free(&tmp);
    av_freep(&avctx->priv_data);
    av_freep(&avctx->internal);
    avctx->codec= NULL;
    goto end;