static int showspectrumpic_request_frame(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    ShowSpectrumContext *s = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];

    int ret;



    ret = ff_request_frame(inlink);

    if (ret == AVERROR_EOF && s->outpicref) {

        int samples = av_audio_fifo_size(s->fifo);

        int consumed = 0;

        int y, x = 0, sz = s->orientation == VERTICAL ? s->w : s->h;

        int ch, spf, spb;

        AVFrame *fin;



        spf = s->win_size * (samples / ((s->win_size * sz) * ceil(samples / (float)(s->win_size * sz))));

        spb = (samples / (spf * sz)) * spf;



        fin = ff_get_audio_buffer(inlink, s->win_size);

        if (!fin)

            return AVERROR(ENOMEM);



        while (x < sz) {

            ret = av_audio_fifo_peek(s->fifo, (void **)fin->extended_data, s->win_size);

            if (ret < 0) {

                av_frame_free(&fin);

                return ret;

            }



            av_audio_fifo_drain(s->fifo, spf);



            if (ret < s->win_size) {

                for (ch = 0; ch < s->nb_display_channels; ch++) {

                    memset(fin->extended_data[ch] + ret * sizeof(float), 0,

                           (s->win_size - ret) * sizeof(float));

                }

            }



            ctx->internal->execute(ctx, run_channel_fft, fin, NULL, s->nb_display_channels);

            acalc_magnitudes(s);



            consumed += spf;

            if (consumed >= spb) {

                int h = s->orientation == VERTICAL ? s->h : s->w;



                scale_magnitudes(s, 1. / (consumed / spf));

                plot_spectrum_column(inlink, fin);

                consumed = 0;

                x++;

                for (ch = 0; ch < s->nb_display_channels; ch++)

                    memset(s->magnitudes[ch], 0, h * sizeof(float));

            }

        }



        av_frame_free(&fin);

        s->outpicref->pts = 0;



        if (s->legend) {

            int multi = (s->mode == SEPARATE && s->color_mode == CHANNEL);

            float spp = samples / (float)sz;

            uint8_t *dst;



            drawtext(s->outpicref, 2, outlink->h - 10, "CREATED BY LIBAVFILTER", 0);



            dst = s->outpicref->data[0] + (s->start_y - 1) * s->outpicref->linesize[0] + s->start_x - 1;

            for (x = 0; x < s->w + 1; x++)

                dst[x] = 200;

            dst = s->outpicref->data[0] + (s->start_y + s->h) * s->outpicref->linesize[0] + s->start_x - 1;

            for (x = 0; x < s->w + 1; x++)

                dst[x] = 200;

            for (y = 0; y < s->h + 2; y++) {

                dst = s->outpicref->data[0] + (y + s->start_y - 1) * s->outpicref->linesize[0];

                dst[s->start_x - 1] = 200;

                dst[s->start_x + s->w] = 200;

            }

            if (s->orientation == VERTICAL) {

                int h = s->mode == SEPARATE ? s->h / s->nb_display_channels : s->h;

                for (ch = 0; ch < (s->mode == SEPARATE ? s->nb_display_channels : 1); ch++) {

                    for (y = 0; y < h; y += 20) {

                        dst = s->outpicref->data[0] + (s->start_y + h * (ch + 1) - y - 1) * s->outpicref->linesize[0];

                        dst[s->start_x - 2] = 200;

                        dst[s->start_x + s->w + 1] = 200;

                    }

                    for (y = 0; y < h; y += 40) {

                        dst = s->outpicref->data[0] + (s->start_y + h * (ch + 1) - y - 1) * s->outpicref->linesize[0];

                        dst[s->start_x - 3] = 200;

                        dst[s->start_x + s->w + 2] = 200;

                    }

                    dst = s->outpicref->data[0] + (s->start_y - 2) * s->outpicref->linesize[0] + s->start_x;

                    for (x = 0; x < s->w; x+=40)

                        dst[x] = 200;

                    dst = s->outpicref->data[0] + (s->start_y - 3) * s->outpicref->linesize[0] + s->start_x;

                    for (x = 0; x < s->w; x+=80)

                        dst[x] = 200;

                    dst = s->outpicref->data[0] + (s->h + s->start_y + 1) * s->outpicref->linesize[0] + s->start_x;

                    for (x = 0; x < s->w; x+=40) {

                        dst[x] = 200;

                    }

                    dst = s->outpicref->data[0] + (s->h + s->start_y + 2) * s->outpicref->linesize[0] + s->start_x;

                    for (x = 0; x < s->w; x+=80) {

                        dst[x] = 200;

                    }

                    for (y = 0; y < h; y += 40) {

                        float hertz = y * (inlink->sample_rate / 2) / (float)(1 << (int)ceil(log2(h)));

                        char *units;



                        if (hertz == 0)

                            units = av_asprintf("DC");

                        else

                            units = av_asprintf("%.2f", hertz);

                        if (!units)

                            return AVERROR(ENOMEM);



                        drawtext(s->outpicref, s->start_x - 8 * strlen(units) - 4, h * (ch + 1) + s->start_y - y - 4, units, 0);

                        av_free(units);

                    }

                }



                for (x = 0; x < s->w; x+=80) {

                    float seconds = x * spp / inlink->sample_rate;

                    char *units;



                    if (x == 0)

                        units = av_asprintf("0");

                    else if (log10(seconds) > 6)

                        units = av_asprintf("%.2fh", seconds / (60 * 60));

                    else if (log10(seconds) > 3)

                        units = av_asprintf("%.2fm", seconds / 60);

                    else

                        units = av_asprintf("%.2fs", seconds);

                    if (!units)

                        return AVERROR(ENOMEM);



                    drawtext(s->outpicref, s->start_x + x - 4 * strlen(units), s->h + s->start_y + 6, units, 0);

                    drawtext(s->outpicref, s->start_x + x - 4 * strlen(units), s->start_y - 12, units, 0);

                    av_free(units);

                }



                drawtext(s->outpicref, outlink->w / 2 - 4 * 4, outlink->h - s->start_y / 2, "TIME", 0);

                drawtext(s->outpicref, s->start_x / 7, outlink->h / 2 - 14 * 4, "FREQUENCY (Hz)", 1);

            } else {

                int w = s->mode == SEPARATE ? s->w / s->nb_display_channels : s->w;

                for (y = 0; y < s->h; y += 20) {

                    dst = s->outpicref->data[0] + (s->start_y + y) * s->outpicref->linesize[0];

                    dst[s->start_x - 2] = 200;

                    dst[s->start_x + s->w + 1] = 200;

                }

                for (y = 0; y < s->h; y += 40) {

                    dst = s->outpicref->data[0] + (s->start_y + y) * s->outpicref->linesize[0];

                    dst[s->start_x - 3] = 200;

                    dst[s->start_x + s->w + 2] = 200;

                }

                for (ch = 0; ch < (s->mode == SEPARATE ? s->nb_display_channels : 1); ch++) {

                    dst = s->outpicref->data[0] + (s->start_y - 2) * s->outpicref->linesize[0] + s->start_x + w * ch;

                    for (x = 0; x < w; x+=40)

                        dst[x] = 200;

                    dst = s->outpicref->data[0] + (s->start_y - 3) * s->outpicref->linesize[0] + s->start_x + w * ch;

                    for (x = 0; x < w; x+=80)

                        dst[x] = 200;

                    dst = s->outpicref->data[0] + (s->h + s->start_y + 1) * s->outpicref->linesize[0] + s->start_x + w * ch;

                    for (x = 0; x < w; x+=40) {

                        dst[x] = 200;

                    }

                    dst = s->outpicref->data[0] + (s->h + s->start_y + 2) * s->outpicref->linesize[0] + s->start_x + w * ch;

                    for (x = 0; x < w; x+=80) {

                        dst[x] = 200;

                    }

                    for (x = 0; x < w; x += 80) {

                        float hertz = x * (inlink->sample_rate / 2) / (float)(1 << (int)ceil(log2(w)));

                        char *units;



                        if (hertz == 0)

                            units = av_asprintf("DC");

                        else

                            units = av_asprintf("%.2f", hertz);

                        if (!units)

                            return AVERROR(ENOMEM);



                        drawtext(s->outpicref, s->start_x - 4 * strlen(units) + x + w * ch, s->start_y - 12, units, 0);

                        drawtext(s->outpicref, s->start_x - 4 * strlen(units) + x + w * ch, s->h + s->start_y + 6, units, 0);

                        av_free(units);

                    }

                }

                for (y = 0; y < s->h; y+=40) {

                    float seconds = y * spp / inlink->sample_rate;

                    char *units;



                    if (x == 0)

                        units = av_asprintf("0");

                    else if (log10(seconds) > 6)

                        units = av_asprintf("%.2fh", seconds / (60 * 60));

                    else if (log10(seconds) > 3)

                        units = av_asprintf("%.2fm", seconds / 60);

                    else

                        units = av_asprintf("%.2fs", seconds);

                    if (!units)

                        return AVERROR(ENOMEM);



                    drawtext(s->outpicref, s->start_x - 8 * strlen(units) - 4, s->start_y + y - 4, units, 0);

                    av_free(units);

                }

                drawtext(s->outpicref, s->start_x / 7, outlink->h / 2 - 4 * 4, "TIME", 1);

                drawtext(s->outpicref, outlink->w / 2 - 14 * 4, outlink->h - s->start_y / 2, "FREQUENCY (Hz)", 0);

            }



            for (ch = 0; ch < (multi ? s->nb_display_channels : 1); ch++) {

                int h = multi ? s->h / s->nb_display_channels : s->h;



                for (y = 0; y < h; y++) {

                    float out[3] = { 0., 127.5, 127.5};

                    int chn;



                    for (chn = 0; chn < (s->mode == SEPARATE ? 1 : s->nb_display_channels); chn++) {

                        float yf, uf, vf;

                        int channel = (multi) ? s->nb_display_channels - ch - 1 : chn;

                        float lout[3];



                        color_range(s, channel, &yf, &uf, &vf);

                        pick_color(s, yf, uf, vf, y / (float)h, lout);

                        out[0] += lout[0];

                        out[1] += lout[1];

                        out[2] += lout[2];

                    }

                    memset(s->outpicref->data[0]+(s->start_y + h * (ch + 1) - y - 1) * s->outpicref->linesize[0] + s->w + s->start_x + 20, av_clip_uint8(out[0]), 10);

                    memset(s->outpicref->data[1]+(s->start_y + h * (ch + 1) - y - 1) * s->outpicref->linesize[1] + s->w + s->start_x + 20, av_clip_uint8(out[1]), 10);

                    memset(s->outpicref->data[2]+(s->start_y + h * (ch + 1) - y - 1) * s->outpicref->linesize[2] + s->w + s->start_x + 20, av_clip_uint8(out[2]), 10);

                }



                for (y = 0; ch == 0 && y < h; y += h / 10) {

                    float value = 120.0 * log10(1. - y / (float)h);

                    char *text;



                    if (value < -120)

                        break;

                    text = av_asprintf("%.0f dB", value);

                    if (!text)

                        continue;

                    drawtext(s->outpicref, s->w + s->start_x + 35, s->start_y + y - 5, text, 0);

                    av_free(text);

                }

            }

        }



        ret = ff_filter_frame(outlink, s->outpicref);

        s->outpicref = NULL;

    }



    return ret;

}
