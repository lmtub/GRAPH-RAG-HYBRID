void Process(void *ctx, AVPicture *picture, enum PixelFormat pix_fmt, int width, int height, int64_t pts)

{

    ContextInfo *ci = (ContextInfo *) ctx;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    int rowsize = picture->linesize[0];



#if 0

    av_log(NULL, AV_LOG_DEBUG, "pix_fmt = %d, width = %d, pts = %lld, ci->next_pts = %lld\n",

        pix_fmt, width, pts, ci->next_pts);

#endif



    if (pts < ci->next_pts)

        return;



    if (width < ci->min_width)

        return;



    ci->next_pts = pts + 1000000;



    if (pix_fmt == PIX_FMT_YUV420P) {

        uint8_t *y, *u, *v;

        int width2 = width >> 1;

        int inrange = 0;

        int pixcnt;

        int h;

        int h_start, h_end;

        int w_start, w_end;



        h_end = 2 * ((ci->inset * height) / 200);

        h_start = height - h_end;



        w_end = (ci->inset * width2) / 100;

        w_start = width2 - w_end;



        pixcnt = ((h_start - h_end) >> 1) * (w_start - w_end);



        y = picture->data[0] + h_end * picture->linesize[0] + w_end * 2;

        u = picture->data[1] + h_end * picture->linesize[1] / 2 + w_end;

        v = picture->data[2] + h_end * picture->linesize[2] / 2 + w_end;



        for (h = h_start; h > h_end; h -= 2) {

            int w;



            for (w = w_start; w > w_end; w--) {

                unsigned int r,g,b;

                HSV hsv;

                int cb, cr, yt, r_add, g_add, b_add;



                YUV_TO_RGB1_CCIR(u[0], v[0]);

                YUV_TO_RGB2_CCIR(r, g, b, y[0]);



                get_hsv(&hsv, r, g, b);



                if (ci->debug > 1)

                    av_log(NULL, AV_LOG_DEBUG, "(%d,%d,%d) -> (%d,%d,%d)\n",

                        r,g,b,hsv.h,hsv.s,hsv.v);





                if (hsv.h >= ci->dark.h && hsv.h <= ci->bright.h &&

                    hsv.s >= ci->dark.s && hsv.s <= ci->bright.s &&

                    hsv.v >= ci->dark.v && hsv.v <= ci->bright.v) {

                    inrange++;

                } else if (ci->zapping) {

                    y[0] = y[1] = y[rowsize] = y[rowsize + 1] = 16;

                    u[0] = 128;

                    v[0] = 128;

                }



                y+= 2;

                u++;

                v++;

            }



            y += picture->linesize[0] * 2 - (w_start - w_end) * 2;

            u += picture->linesize[1] - (w_start - w_end);

            v += picture->linesize[2] - (w_start - w_end);

        }



        if (ci->debug)

            av_log(NULL, AV_LOG_INFO, "Fish: Inrange=%d of %d = %d threshold\n", inrange, pixcnt, 1000 * inrange / pixcnt);



        if (inrange * 1000 / pixcnt >= ci->threshold) {

            /* Save to file */

            int size;

            char *buf;

            AVPicture picture1;

            static int frame_counter;

            static int foundfile;



            if ((frame_counter++ % 20) == 0) {

                /* Check how many files we have */

                DIR *d;



                foundfile = 0;



                d = opendir(ci->dir);

                if (d) {

                    struct dirent *dent;



                    while ((dent = readdir(d))) {

                        if (strncmp("fishimg", dent->d_name, 7) == 0) {

                            if (strcmp(".ppm", dent->d_name + strlen(dent->d_name) - 4) == 0) {

                                foundfile++;

                            }

                        }

                    }

                    closedir(d);

                }

            }



            if (foundfile < ci->file_limit) {

                FILE *f;

                char fname[256];



                size = avpicture_get_size(PIX_FMT_RGB24, width, height);

                buf = av_malloc(size);



                avpicture_fill(&picture1, buf, PIX_FMT_RGB24, width, height);



                // if we already got a SWS context, let's realloc if is not re-useable

                ci->toRGB_convert_ctx = sws_getCachedContext(ci->toRGB_convert_ctx,

                                            width, height, pix_fmt,

                                            width, height, PIX_FMT_RGB24,

                                            sws_flags, NULL, NULL, NULL);

                if (ci->toRGB_convert_ctx == NULL) {

                    av_log(NULL, AV_LOG_ERROR,

                           "Cannot initialize the toRGB conversion context\n");

                    return;

                }

                // img_convert parameters are          2 first destination, then 4 source

                // sws_scale   parameters are context, 4 first source,      then 2 destination

                sws_scale(ci->toRGB_convert_ctx,

                              picture->data, picture->linesize, 0, height,

                              picture1.data, picture1.linesize);



                    /* Write out the PPM file */

                    snprintf(fname, sizeof(fname), "%s/fishimg%ld_%"PRId64".ppm", ci->dir, (long)(av_gettime() / 1000000), pts);

                    f = fopen(fname, "w");

                    if (f) {

                        fprintf(f, "P6 %d %d 255\n", width, height);

                        fwrite(buf, width * height * 3, 1, f);

                        fclose(f);

                    }



                av_free(buf);

                ci->next_pts = pts + ci->min_interval;

            }

        }

    }

}
