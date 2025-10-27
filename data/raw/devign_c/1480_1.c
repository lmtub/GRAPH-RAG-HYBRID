static int cng_decode_frame(AVCodecContext *avctx, void *data,
                            int *got_frame_ptr, AVPacket *avpkt)
{
    AVFrame *frame = data;
    CNGContext *p = avctx->priv_data;
    int buf_size  = avpkt->size;
    int ret, i;
    int16_t *buf_out;
    float e = 1.0;
    float scaling;
    if (avpkt->size) {
        int dbov = -avpkt->data[0];
        p->target_energy = 1081109975 * ff_exp10(dbov / 10.0) * 0.75;
        memset(p->target_refl_coef, 0, p->order * sizeof(*p->target_refl_coef));
        for (i = 0; i < FFMIN(avpkt->size - 1, p->order); i++) {
            p->target_refl_coef[i] = (avpkt->data[1 + i] - 127) / 128.0;
    if (p->inited) {
        p->energy = p->energy / 2 + p->target_energy / 2;
        for (i = 0; i < p->order; i++)
            p->refl_coef[i] = 0.6 *p->refl_coef[i] + 0.4 * p->target_refl_coef[i];
    } else {
        p->energy = p->target_energy;
        memcpy(p->refl_coef, p->target_refl_coef, p->order * sizeof(*p->refl_coef));
        p->inited = 1;
    make_lpc_coefs(p->lpc_coef, p->refl_coef, p->order);
    for (i = 0; i < p->order; i++)
        e *= 1.0 - p->refl_coef[i]*p->refl_coef[i];
    scaling = sqrt(e * p->energy / 1081109975);
    for (i = 0; i < avctx->frame_size; i++) {
        int r = (av_lfg_get(&p->lfg) & 0xffff) - 0x8000;
        p->excitation[i] = scaling * r;
    ff_celp_lp_synthesis_filterf(p->filter_out + p->order, p->lpc_coef,
                                 p->excitation, avctx->frame_size, p->order);
    frame->nb_samples = avctx->frame_size;
    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
        return ret;
    buf_out = (int16_t *)frame->data[0];
    for (i = 0; i < avctx->frame_size; i++)
        buf_out[i] = p->filter_out[i + p->order];
    memcpy(p->filter_out, p->filter_out + avctx->frame_size,
           p->order * sizeof(*p->filter_out));
    *got_frame_ptr = 1;
    return buf_size;