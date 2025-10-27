static int plot_cqt(AVFilterContext *ctx)

{

    AVFilterLink *outlink = ctx->outputs[0];

    ShowCQTContext *s = ctx->priv;

    int ret;



    memcpy(s->fft_result, s->fft_data, s->fft_len * sizeof(*s->fft_data));

    av_fft_permute(s->fft_ctx, s->fft_result);

    av_fft_calc(s->fft_ctx, s->fft_result);

    s->fft_result[s->fft_len] = s->fft_result[0];

    s->cqt_calc(s->cqt_result, s->fft_result, s->coeffs, s->cqt_len, s->fft_len);

    process_cqt(s);

    if (s->sono_h)

        s->update_sono(s->sono_frame, s->c_buf, s->sono_idx);

    if (!s->sono_count) {

        AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

        if (!out)

            return AVERROR(ENOMEM);

        if (s->bar_h)

            s->draw_bar(out, s->h_buf, s->rcp_h_buf, s->c_buf, s->bar_h);

        if (s->axis_h)

            s->draw_axis(out, s->axis_frame, s->c_buf, s->bar_h);

        if (s->sono_h)

            s->draw_sono(out, s->sono_frame, s->bar_h + s->axis_h, s->sono_idx);

        out->pts = s->frame_count;

        ret = ff_filter_frame(outlink, out);

        s->frame_count++;

    }

    s->sono_count = (s->sono_count + 1) % s->count;

    if (s->sono_h)

        s->sono_idx = (s->sono_idx + s->sono_h - 1) % s->sono_h;

    return ret;

}
