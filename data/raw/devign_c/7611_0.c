void Process(void *ctx, AVPicture *picture, enum PixelFormat pix_fmt, int width, int height, INT64 pts)

{

    ContextInfo *ci = (ContextInfo *) ctx;

    AVPicture picture1;

    Imlib_Image image;

    DATA32 *data;



    image = get_cached_image(ci, width, height);



    if (!image) {

        image = imlib_create_image(width, height);

        put_cached_image(ci, image, width, height);

    }



    imlib_context_set_image(image);

    data = imlib_image_get_data();



    if (pix_fmt != PIX_FMT_RGBA32) {

        avpicture_fill(&picture1, (UINT8 *) data, PIX_FMT_RGBA32, width, height);

        if (img_convert(&picture1, PIX_FMT_RGBA32, 

                        picture, pix_fmt, width, height) < 0) {

            goto done;

        }

    } else {

        av_abort();

    }



    imlib_image_set_has_alpha(0);



    {

        int wid, hig, h_a, v_a;                                                   

        char buff[1000];

        char tbuff[1000];

        char *tbp = ci->text;

        time_t now = time(0);

        char *p, *q;

        int x, y;



        if (ci->file) {

            int fd = open(ci->file, O_RDONLY);



            if (fd < 0) {

                tbp = "[File not found]";

            } else {

                int l = read(fd, tbuff, sizeof(tbuff) - 1);



                if (l >= 0) {

                    tbuff[l] = 0;

                    tbp = tbuff;

                } else {

                    tbp = "[I/O Error]";

                }

                close(fd);

            }

        }



        strftime(buff, sizeof(buff), tbp, localtime(&now));



        x = ci->x;

        y = ci->y;



        for (p = buff; p; p = q) {

            q = strchr(p, '\n');

            if (q)

                *q++ = 0;



            imlib_text_draw_with_return_metrics(x, y, p, &wid, &hig, &h_a, &v_a);

            y += v_a;

        }

    }



    if (pix_fmt != PIX_FMT_RGBA32) {

        if (img_convert(picture, pix_fmt, 

                        &picture1, PIX_FMT_RGBA32, width, height) < 0) {

        }

    }



done:

    ;

}
