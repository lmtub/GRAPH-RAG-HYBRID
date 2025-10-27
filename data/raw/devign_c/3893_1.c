void ff_check_pixfmt_descriptors(void){

    int i, j;



    for (i=0; i<FF_ARRAY_ELEMS(av_pix_fmt_descriptors); i++) {

        const AVPixFmtDescriptor *d = &av_pix_fmt_descriptors[i];

        uint8_t fill[4][8+6+3] = {{0}};

        uint8_t *data[4] = {fill[0], fill[1], fill[2], fill[3]};

        int linesize[4] = {0,0,0,0};

        uint16_t tmp[2];



        if (!d->name && !d->nb_components && !d->log2_chroma_w && !d->log2_chroma_h && !d->flags)

            continue;

//         av_log(NULL, AV_LOG_DEBUG, "Checking: %s\n", d->name);

        av_assert0(d->log2_chroma_w <= 3);

        av_assert0(d->log2_chroma_h <= 3);

        av_assert0(d->nb_components <= 4);

        av_assert0(d->name && d->name[0]);

        av_assert0((d->nb_components==4 || d->nb_components==2) == !!(d->flags & AV_PIX_FMT_FLAG_ALPHA));

        av_assert2(av_get_pix_fmt(d->name) == i);



        for (j=0; j<FF_ARRAY_ELEMS(d->comp); j++) {

            const AVComponentDescriptor *c = &d->comp[j];

            if(j>=d->nb_components) {

                av_assert0(!c->plane && !c->step_minus1 && !c->offset_plus1 && !c->shift && !c->depth_minus1);

                continue;

            }

            if (d->flags & AV_PIX_FMT_FLAG_BITSTREAM) {

                av_assert0(c->step_minus1 >= c->depth_minus1);

            } else {

                av_assert0(8*(c->step_minus1+1) >= c->depth_minus1+1);

            }

            av_read_image_line(tmp, (void*)data, linesize, d, 0, 0, j, 2, 0);

            if (!strncmp(d->name, "bayer_", 6))

                continue;

            av_assert0(tmp[0] == 0 && tmp[1] == 0);

            tmp[0] = tmp[1] = (1<<(c->depth_minus1 + 1)) - 1;

            av_write_image_line(tmp, data, linesize, d, 0, 0, j, 2);

        }

    }

}
