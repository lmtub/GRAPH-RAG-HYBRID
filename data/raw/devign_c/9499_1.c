static int vp8_decode_frame_header(VP8Context *s, const uint8_t *buf, int buf_size)
{
    VP56RangeCoder *c = &s->c;
    int header_size, hscale, vscale, ret;
    int width  = s->avctx->width;
    int height = s->avctx->height;
    s->keyframe  = !(buf[0] & 1);
    s->profile   =  (buf[0]>>1) & 7;
    s->invisible = !(buf[0] & 0x10);
    header_size  = AV_RL24(buf) >> 5;
    buf      += 3;
    buf_size -= 3;
    if (s->profile > 3)
        av_log(s->avctx, AV_LOG_WARNING, "Unknown profile %d\n", s->profile);
    if (!s->profile)
        memcpy(s->put_pixels_tab, s->vp8dsp.put_vp8_epel_pixels_tab,
               sizeof(s->put_pixels_tab));
    else    // profile 1-3 use bilinear, 4+ aren't defined so whatever
        memcpy(s->put_pixels_tab, s->vp8dsp.put_vp8_bilinear_pixels_tab,
               sizeof(s->put_pixels_tab));
    if (header_size > buf_size - 7 * s->keyframe) {
        av_log(s->avctx, AV_LOG_ERROR, "Header size larger than data provided\n");
    if (s->keyframe) {
        if (AV_RL24(buf) != 0x2a019d) {
            av_log(s->avctx, AV_LOG_ERROR,
                   "Invalid start code 0x%x\n", AV_RL24(buf));
        width     = AV_RL16(buf + 3) & 0x3fff;
        height    = AV_RL16(buf + 5) & 0x3fff;
        hscale    = buf[4] >> 6;
        vscale    = buf[6] >> 6;
        buf      += 7;
        buf_size -= 7;
        if (hscale || vscale)
            avpriv_request_sample(s->avctx, "Upscaling");
        s->update_golden = s->update_altref = VP56_FRAME_CURRENT;
        vp78_reset_probability_tables(s);
        memcpy(s->prob->pred16x16, vp8_pred16x16_prob_inter,
               sizeof(s->prob->pred16x16));
        memcpy(s->prob->pred8x8c, vp8_pred8x8c_prob_inter,
               sizeof(s->prob->pred8x8c));
        memcpy(s->prob->mvc, vp8_mv_default_prob,
               sizeof(s->prob->mvc));
        memset(&s->segmentation, 0, sizeof(s->segmentation));
        memset(&s->lf_delta, 0, sizeof(s->lf_delta));
    ff_vp56_init_range_decoder(c, buf, header_size);
    buf      += header_size;
    buf_size -= header_size;
    if (s->keyframe) {
        s->colorspace = vp8_rac_get(c);
        if (s->colorspace)
            av_log(s->avctx, AV_LOG_WARNING, "Unspecified colorspace\n");
        s->fullrange = vp8_rac_get(c);
    if ((s->segmentation.enabled = vp8_rac_get(c)))
        parse_segment_info(s);
    else
        s->segmentation.update_map = 0; // FIXME: move this to some init function?
    s->filter.simple    = vp8_rac_get(c);
    s->filter.level     = vp8_rac_get_uint(c, 6);
    s->filter.sharpness = vp8_rac_get_uint(c, 3);
    if ((s->lf_delta.enabled = vp8_rac_get(c)))
        if (vp8_rac_get(c))
            update_lf_deltas(s);
    if (setup_partitions(s, buf, buf_size)) {
        av_log(s->avctx, AV_LOG_ERROR, "Invalid partitions\n");
    if (!s->macroblocks_base || /* first frame */
        width != s->avctx->width || height != s->avctx->height ||
        (width+15)/16 != s->mb_width || (height+15)/16 != s->mb_height)
        if ((ret = vp8_update_dimensions(s, width, height)) < 0)
            return ret;
    vp8_get_quants(s);
    if (!s->keyframe) {
        update_refs(s);
        s->sign_bias[VP56_FRAME_GOLDEN]               = vp8_rac_get(c);
        s->sign_bias[VP56_FRAME_GOLDEN2 /* altref */] = vp8_rac_get(c);
    // if we aren't saving this frame's probabilities for future frames,
    // make a copy of the current probabilities
    if (!(s->update_probabilities = vp8_rac_get(c)))
        s->prob[1] = s->prob[0];
    s->update_last = s->keyframe || vp8_rac_get(c);
    vp78_update_probability_tables(s);
    if ((s->mbskip_enabled = vp8_rac_get(c)))
        s->prob->mbskip = vp8_rac_get_uint(c, 8);
    if (!s->keyframe) {
        s->prob->intra  = vp8_rac_get_uint(c, 8);
        s->prob->last   = vp8_rac_get_uint(c, 8);
        s->prob->golden = vp8_rac_get_uint(c, 8);
        vp78_update_pred16x16_pred8x8_mvc_probabilities(s, VP8_MVC_SIZE);
    return 0;