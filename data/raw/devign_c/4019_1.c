int attribute_align_arg avcodec_open2(AVCodecContext *avctx, const AVCodec *codec, AVDictionary **options)
{
    int ret = 0;
    AVDictionary *tmp = NULL;
    if (avcodec_is_open(avctx))
        return 0;
    if ((!codec && !avctx->codec)) {
        av_log(avctx, AV_LOG_ERROR, "No codec provided to avcodec_open2().\n");
        return AVERROR(EINVAL);
    if ((codec && avctx->codec && codec != avctx->codec)) {
        av_log(avctx, AV_LOG_ERROR, "This AVCodecContext was allocated for %s, "
                                    "but %s passed to avcodec_open2().\n", avctx->codec->name, codec->name);
        return AVERROR(EINVAL);
    if (!codec)
        codec = avctx->codec;
    if (avctx->extradata_size < 0 || avctx->extradata_size >= FF_MAX_EXTRADATA_SIZE)
        return AVERROR(EINVAL);
    if (options)
        av_dict_copy(&tmp, *options, 0);
    /* If there is a user-supplied mutex locking routine, call it. */
    if (!(codec->caps_internal & FF_CODEC_CAP_INIT_THREADSAFE) && codec->init) {
        if (lockmgr_cb) {
            if ((*lockmgr_cb)(&codec_mutex, AV_LOCK_OBTAIN))
                return -1;
        entangled_thread_counter++;
        if (entangled_thread_counter != 1) {
            av_log(avctx, AV_LOG_ERROR,
                   "Insufficient thread locking. At least %d threads are "
                   "calling avcodec_open2() at the same time right now.\n",
                   entangled_thread_counter);
            ret = -1;
            goto end;
    avctx->internal = av_mallocz(sizeof(AVCodecInternal));
    if (!avctx->internal) {
        ret = AVERROR(ENOMEM);
        goto end;
    avctx->internal->pool = av_mallocz(sizeof(*avctx->internal->pool));
    if (!avctx->internal->pool) {
        ret = AVERROR(ENOMEM);
    avctx->internal->to_free = av_frame_alloc();
    if (!avctx->internal->to_free) {
        ret = AVERROR(ENOMEM);
    avctx->internal->buffer_frame = av_frame_alloc();
    if (!avctx->internal->buffer_frame) {
        ret = AVERROR(ENOMEM);
    avctx->internal->buffer_pkt = av_packet_alloc();
    if (!avctx->internal->buffer_pkt) {
        ret = AVERROR(ENOMEM);
    if (codec->priv_data_size > 0) {
        if (!avctx->priv_data) {
            avctx->priv_data = av_mallocz(codec->priv_data_size);
            if (!avctx->priv_data) {
                ret = AVERROR(ENOMEM);
                goto end;
            if (codec->priv_class) {
                *(const AVClass **)avctx->priv_data = codec->priv_class;
                av_opt_set_defaults(avctx->priv_data);
        if (codec->priv_class && (ret = av_opt_set_dict(avctx->priv_data, &tmp)) < 0)
    } else {
        avctx->priv_data = NULL;
    if ((ret = av_opt_set_dict(avctx, &tmp)) < 0)
    if (avctx->coded_width && avctx->coded_height && !avctx->width && !avctx->height)
        ret = ff_set_dimensions(avctx, avctx->coded_width, avctx->coded_height);
    else if (avctx->width && avctx->height)
        ret = ff_set_dimensions(avctx, avctx->width, avctx->height);
    if (ret < 0)
    if ((avctx->coded_width || avctx->coded_height || avctx->width || avctx->height)
        && (  av_image_check_size(avctx->coded_width, avctx->coded_height, 0, avctx) < 0
           || av_image_check_size(avctx->width,       avctx->height,       0, avctx) < 0)) {
        av_log(avctx, AV_LOG_WARNING, "ignoring invalid width/height values\n");
        ff_set_dimensions(avctx, 0, 0);
    if (avctx->width > 0 && avctx->height > 0) {
        if (av_image_check_sar(avctx->width, avctx->height,
                               avctx->sample_aspect_ratio) < 0) {
            av_log(avctx, AV_LOG_WARNING, "ignoring invalid SAR: %u/%u\n",
                   avctx->sample_aspect_ratio.num,
                   avctx->sample_aspect_ratio.den);
            avctx->sample_aspect_ratio = (AVRational){ 0, 1 };
    /* if the decoder init function was already called previously,
     * free the already allocated subtitle_header before overwriting it */
    if (av_codec_is_decoder(codec))
        av_freep(&avctx->subtitle_header);
    if (avctx->channels > FF_SANE_NB_CHANNELS) {
    avctx->codec = codec;
    if ((avctx->codec_type == AVMEDIA_TYPE_UNKNOWN || avctx->codec_type == codec->type) &&
        avctx->codec_id == AV_CODEC_ID_NONE) {
        avctx->codec_type = codec->type;
        avctx->codec_id   = codec->id;
    if (avctx->codec_id != codec->id || (avctx->codec_type != codec->type
                                         && avctx->codec_type != AVMEDIA_TYPE_ATTACHMENT)) {
        av_log(avctx, AV_LOG_ERROR, "codec type or id mismatches\n");
    avctx->frame_number = 0;
    if ((avctx->codec->capabilities & AV_CODEC_CAP_EXPERIMENTAL) &&
        avctx->strict_std_compliance > FF_COMPLIANCE_EXPERIMENTAL) {
        ret = AVERROR_EXPERIMENTAL;
    if (avctx->codec_type == AVMEDIA_TYPE_AUDIO &&
        (!avctx->time_base.num || !avctx->time_base.den)) {
        avctx->time_base.num = 1;
        avctx->time_base.den = avctx->sample_rate;
    if (HAVE_THREADS) {
        ret = ff_thread_init(avctx);
        if (ret < 0) {
    if (!HAVE_THREADS && !(codec->capabilities & AV_CODEC_CAP_AUTO_THREADS))
        avctx->thread_count = 1;
    if (av_codec_is_encoder(avctx->codec)) {
        int i;
#if FF_API_CODED_FRAME
FF_DISABLE_DEPRECATION_WARNINGS
        avctx->coded_frame = av_frame_alloc();
        if (!avctx->coded_frame) {
            ret = AVERROR(ENOMEM);
FF_ENABLE_DEPRECATION_WARNINGS
#endif
        if (avctx->codec->sample_fmts) {
            for (i = 0; avctx->codec->sample_fmts[i] != AV_SAMPLE_FMT_NONE; i++) {
                if (avctx->sample_fmt == avctx->codec->sample_fmts[i])
                    break;
                if (avctx->channels == 1 &&
                    av_get_planar_sample_fmt(avctx->sample_fmt) ==
                    av_get_planar_sample_fmt(avctx->codec->sample_fmts[i])) {
                    avctx->sample_fmt = avctx->codec->sample_fmts[i];
                    break;
            if (avctx->codec->sample_fmts[i] == AV_SAMPLE_FMT_NONE) {
                av_log(avctx, AV_LOG_ERROR, "Specified sample_fmt is not supported.\n");
        if (avctx->codec->pix_fmts) {
            for (i = 0; avctx->codec->pix_fmts[i] != AV_PIX_FMT_NONE; i++)
                if (avctx->pix_fmt == avctx->codec->pix_fmts[i])
                    break;
            if (avctx->codec->pix_fmts[i] == AV_PIX_FMT_NONE) {
                av_log(avctx, AV_LOG_ERROR, "Specified pix_fmt is not supported\n");
            if (avctx->codec->pix_fmts[i] == AV_PIX_FMT_YUVJ420P ||
                avctx->codec->pix_fmts[i] == AV_PIX_FMT_YUVJ422P ||
                avctx->codec->pix_fmts[i] == AV_PIX_FMT_YUVJ440P ||
                avctx->codec->pix_fmts[i] == AV_PIX_FMT_YUVJ444P)
                avctx->color_range = AVCOL_RANGE_JPEG;
        if (avctx->codec->supported_samplerates) {
            for (i = 0; avctx->codec->supported_samplerates[i] != 0; i++)
                if (avctx->sample_rate == avctx->codec->supported_samplerates[i])
                    break;
            if (avctx->codec->supported_samplerates[i] == 0) {
                av_log(avctx, AV_LOG_ERROR, "Specified sample_rate is not supported\n");
        if (avctx->codec->channel_layouts) {
            if (!avctx->channel_layout) {
                av_log(avctx, AV_LOG_WARNING, "channel_layout not specified\n");
            } else {
                for (i = 0; avctx->codec->channel_layouts[i] != 0; i++)
                    if (avctx->channel_layout == avctx->codec->channel_layouts[i])
                        break;
                if (avctx->codec->channel_layouts[i] == 0) {
                    av_log(avctx, AV_LOG_ERROR, "Specified channel_layout is not supported\n");
        if (avctx->channel_layout && avctx->channels) {
            if (av_get_channel_layout_nb_channels(avctx->channel_layout) != avctx->channels) {
                av_log(avctx, AV_LOG_ERROR, "channel layout does not match number of channels\n");
        } else if (avctx->channel_layout) {
            avctx->channels = av_get_channel_layout_nb_channels(avctx->channel_layout);
        if (!avctx->rc_initial_buffer_occupancy)
            avctx->rc_initial_buffer_occupancy = avctx->rc_buffer_size * 3 / 4;
        if (avctx->ticks_per_frame &&
            avctx->ticks_per_frame > INT_MAX / avctx->time_base.num) {
            av_log(avctx, AV_LOG_ERROR,
                   "ticks_per_frame %d too large for the timebase %d/%d.",
                   avctx->ticks_per_frame,
                   avctx->time_base.num,
                   avctx->time_base.den);
        if (avctx->hw_frames_ctx) {
            AVHWFramesContext *frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
            if (frames_ctx->format != avctx->pix_fmt) {
                av_log(avctx, AV_LOG_ERROR,
                       "Mismatching AVCodecContext.pix_fmt and AVHWFramesContext.format\n");
    if (avctx->codec->init && !(avctx->active_thread_type & FF_THREAD_FRAME)) {
        ret = avctx->codec->init(avctx);
        if (ret < 0) {
#if FF_API_AUDIOENC_DELAY
    if (av_codec_is_encoder(avctx->codec))
        avctx->delay = avctx->initial_padding;
#endif
    if (av_codec_is_decoder(avctx->codec)) {
        /* validate channel layout from the decoder */
        if (avctx->channel_layout) {
            int channels = av_get_channel_layout_nb_channels(avctx->channel_layout);
            if (!avctx->channels)
                avctx->channels = channels;
            else if (channels != avctx->channels) {
                av_log(avctx, AV_LOG_WARNING,
                       "channel layout does not match number of channels\n");
                avctx->channel_layout = 0;
        if (avctx->channels && avctx->channels < 0 ||
            avctx->channels > FF_SANE_NB_CHANNELS) {
#if FF_API_AVCTX_TIMEBASE
        if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
            avctx->time_base = av_inv_q(avctx->framerate);
#endif
end:
    if (!(codec->caps_internal & FF_CODEC_CAP_INIT_THREADSAFE) && codec->init) {
        entangled_thread_counter--;
        /* Release any user-supplied mutex. */
        if (lockmgr_cb) {
            (*lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE);
    if (options) {
        av_dict_free(options);
        *options = tmp;
    return ret;
free_and_end:
    if (avctx->codec &&
        (avctx->codec->caps_internal & FF_CODEC_CAP_INIT_CLEANUP))
        avctx->codec->close(avctx);
    if (avctx->priv_data && avctx->codec && avctx->codec->priv_class)
        av_opt_free(avctx->priv_data);
    av_opt_free(avctx);
#if FF_API_CODED_FRAME
FF_DISABLE_DEPRECATION_WARNINGS
    av_frame_free(&avctx->coded_frame);
FF_ENABLE_DEPRECATION_WARNINGS
#endif
    av_dict_free(&tmp);
    av_freep(&avctx->priv_data);
    if (avctx->internal) {
        av_frame_free(&avctx->internal->to_free);
        av_freep(&avctx->internal->pool);
    av_freep(&avctx->internal);
    avctx->codec = NULL;
    goto end;