static int plot_cqt(AVFilterLink *inlink)

{

    AVFilterContext *ctx = inlink->dst;

    ShowCQTContext *s = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    int fft_len = 1 << s->fft_bits;

    FFTSample result[VIDEO_WIDTH][4];

    int x, y, ret = 0;

    int linesize = s->outpicref->linesize[0];

    int video_scale = s->fullhd ? 2 : 1;

    int video_width = (VIDEO_WIDTH/2) * video_scale;

    int spectogram_height = (SPECTOGRAM_HEIGHT/2) * video_scale;

    int spectogram_start = (SPECTOGRAM_START/2) * video_scale;

    int font_height = (FONT_HEIGHT/2) * video_scale;



    /* real part contains left samples, imaginary part contains right samples */

    memcpy(s->fft_result, s->fft_data, fft_len * sizeof(*s->fft_data));

    av_fft_permute(s->fft_context, s->fft_result);

    av_fft_calc(s->fft_context, s->fft_result);

    s->fft_result[fft_len] = s->fft_result[0];



    /* calculating cqt */

    for (x = 0; x < VIDEO_WIDTH; x++) {

        int u;

        FFTComplex v = {0,0};

        FFTComplex w = {0,0};

        FFTComplex l, r;



        for (u = 0; u < s->coeffs[x].len; u++) {

            FFTSample value = s->coeffs[x].values[u];

            int index = s->coeffs[x].start + u;

            v.re += value * s->fft_result[index].re;

            v.im += value * s->fft_result[index].im;

            w.re += value * s->fft_result[fft_len - index].re;

            w.im += value * s->fft_result[fft_len - index].im;

        }



        /* separate left and right, (and multiply by 2.0) */

        l.re = v.re + w.re;

        l.im = v.im - w.im;

        r.re = w.im + v.im;

        r.im = w.re - v.re;

        /* result is power, not amplitude */

        result[x][0] = l.re * l.re + l.im * l.im;

        result[x][2] = r.re * r.re + r.im * r.im;

        result[x][1] = 0.5f * (result[x][0] + result[x][2]);



        if (s->gamma2 == 1.0f)

            result[x][3] = result[x][1];

        else if (s->gamma2 == 2.0f)

            result[x][3] = sqrtf(result[x][1]);

        else if (s->gamma2 == 3.0f)

            result[x][3] = cbrtf(result[x][1]);

        else if (s->gamma2 == 4.0f)

            result[x][3] = sqrtf(sqrtf(result[x][1]));

        else

            result[x][3] = expf(logf(result[x][1]) * (1.0f / s->gamma2));



        result[x][0] = FFMIN(1.0f, result[x][0]);

        result[x][1] = FFMIN(1.0f, result[x][1]);

        result[x][2] = FFMIN(1.0f, result[x][2]);

        if (s->gamma == 1.0f) {

            result[x][0] = 255.0f * result[x][0];

            result[x][1] = 255.0f * result[x][1];

            result[x][2] = 255.0f * result[x][2];

        } else if (s->gamma == 2.0f) {

            result[x][0] = 255.0f * sqrtf(result[x][0]);

            result[x][1] = 255.0f * sqrtf(result[x][1]);

            result[x][2] = 255.0f * sqrtf(result[x][2]);

        } else if (s->gamma == 3.0f) {

            result[x][0] = 255.0f * cbrtf(result[x][0]);

            result[x][1] = 255.0f * cbrtf(result[x][1]);

            result[x][2] = 255.0f * cbrtf(result[x][2]);

        } else if (s->gamma == 4.0f) {

            result[x][0] = 255.0f * sqrtf(sqrtf(result[x][0]));

            result[x][1] = 255.0f * sqrtf(sqrtf(result[x][1]));

            result[x][2] = 255.0f * sqrtf(sqrtf(result[x][2]));

        } else {

            result[x][0] = 255.0f * expf(logf(result[x][0]) * (1.0f / s->gamma));

            result[x][1] = 255.0f * expf(logf(result[x][1]) * (1.0f / s->gamma));

            result[x][2] = 255.0f * expf(logf(result[x][2]) * (1.0f / s->gamma));

        }

    }



    if (!s->fullhd) {

        for (x = 0; x < video_width; x++) {

            result[x][0] = 0.5f * (result[2*x][0] + result[2*x+1][0]);

            result[x][1] = 0.5f * (result[2*x][1] + result[2*x+1][1]);

            result[x][2] = 0.5f * (result[2*x][2] + result[2*x+1][2]);

            result[x][3] = 0.5f * (result[2*x][3] + result[2*x+1][3]);

        }

    }



    for (x = 0; x < video_width; x++) {

        s->spectogram[s->spectogram_index*linesize + 3*x] = result[x][0] + 0.5f;

        s->spectogram[s->spectogram_index*linesize + 3*x + 1] = result[x][1] + 0.5f;

        s->spectogram[s->spectogram_index*linesize + 3*x + 2] = result[x][2] + 0.5f;

    }



    /* drawing */

    if (!s->spectogram_count) {

        uint8_t *data = (uint8_t*) s->outpicref->data[0];

        float rcp_result[VIDEO_WIDTH];

        int total_length = linesize * spectogram_height;

        int back_length = linesize * s->spectogram_index;



        for (x = 0; x < video_width; x++)

            rcp_result[x] = 1.0f / (result[x][3]+0.0001f);



        /* drawing bar */

        for (y = 0; y < spectogram_height; y++) {

            float height = (spectogram_height - y) * (1.0f/spectogram_height);

            uint8_t *lineptr = data + y * linesize;

            for (x = 0; x < video_width; x++) {

                float mul;

                if (result[x][3] <= height) {

                    *lineptr++ = 0;

                    *lineptr++ = 0;

                    *lineptr++ = 0;

                } else {

                    mul = (result[x][3] - height) * rcp_result[x];

                    *lineptr++ = mul * result[x][0] + 0.5f;

                    *lineptr++ = mul * result[x][1] + 0.5f;

                    *lineptr++ = mul * result[x][2] + 0.5f;

                }

            }

        }



        /* drawing font */

        if (s->font_alpha && s->draw_text) {

            for (y = 0; y < font_height; y++) {

                uint8_t *lineptr = data + (spectogram_height + y) * linesize;

                uint8_t *spectogram_src = s->spectogram + s->spectogram_index * linesize;

                uint8_t *fontcolor_value = s->fontcolor_value;

                for (x = 0; x < video_width; x++) {

                    uint8_t alpha = s->font_alpha[y*video_width+x];

                    lineptr[3*x] = (spectogram_src[3*x] * (255-alpha) + fontcolor_value[0] * alpha + 255) >> 8;

                    lineptr[3*x+1] = (spectogram_src[3*x+1] * (255-alpha) + fontcolor_value[1] * alpha + 255) >> 8;

                    lineptr[3*x+2] = (spectogram_src[3*x+2] * (255-alpha) + fontcolor_value[2] * alpha + 255) >> 8;

                    fontcolor_value += 3;

                }

            }

        } else if (s->draw_text) {

            for (y = 0; y < font_height; y++) {

                uint8_t *lineptr = data + (spectogram_height + y) * linesize;

                memcpy(lineptr, s->spectogram + s->spectogram_index * linesize, video_width*3);

            }

            for (x = 0; x < video_width; x += video_width/10) {

                int u;

                static const char str[] = "EF G A BC D ";

                uint8_t *startptr = data + spectogram_height * linesize + x * 3;

                for (u = 0; str[u]; u++) {

                    int v;

                    for (v = 0; v < 16; v++) {

                        uint8_t *p = startptr + v * linesize * video_scale + 8 * 3 * u * video_scale;

                        int ux = x + 8 * u * video_scale;

                        int mask;

                        for (mask = 0x80; mask; mask >>= 1) {

                            if (mask & avpriv_vga16_font[str[u] * 16 + v]) {

                                p[0] = s->fontcolor_value[3*ux];

                                p[1] = s->fontcolor_value[3*ux+1];

                                p[2] = s->fontcolor_value[3*ux+2];

                                if (video_scale == 2) {

                                    p[linesize] = p[0];

                                    p[linesize+1] = p[1];

                                    p[linesize+2] = p[2];

                                    p[3] = p[linesize+3] = s->fontcolor_value[3*ux+3];

                                    p[4] = p[linesize+4] = s->fontcolor_value[3*ux+4];

                                    p[5] = p[linesize+5] = s->fontcolor_value[3*ux+5];

                                }

                            }

                            p  += 3 * video_scale;

                            ux += video_scale;

                        }

                    }

                }

            }

        } else {

            for (y = 0; y < font_height; y++) {

                uint8_t *lineptr = data + (spectogram_height + y) * linesize;

                uint8_t *spectogram_src = s->spectogram + s->spectogram_index * linesize;

                for (x = 0; x < video_width; x++) {

                    lineptr[3*x] = spectogram_src[3*x];

                    lineptr[3*x+1] = spectogram_src[3*x+1];

                    lineptr[3*x+2] = spectogram_src[3*x+2];

                }

            }

        }



        /* drawing spectogram/sonogram */

        data += spectogram_start * linesize;

        memcpy(data, s->spectogram + s->spectogram_index*linesize, total_length - back_length);



        data += total_length - back_length;

        if (back_length)

            memcpy(data, s->spectogram, back_length);



        s->outpicref->pts = s->frame_count;

        ret = ff_filter_frame(outlink, av_frame_clone(s->outpicref));

        s->req_fullfilled = 1;

        s->frame_count++;

    }

    s->spectogram_count = (s->spectogram_count + 1) % s->count;

    s->spectogram_index = (s->spectogram_index + spectogram_height - 1) % spectogram_height;

    return ret;

}
