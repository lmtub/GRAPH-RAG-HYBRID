static int dirac_decode_frame_internal(DiracContext *s)

{

    DWTContext d;

    int y, i, comp, dsty;

    int ret;



    if (s->low_delay) {

        /* [DIRAC_STD] 13.5.1 low_delay_transform_data() */

        for (comp = 0; comp < 3; comp++) {

            Plane *p = &s->plane[comp];

            memset(p->idwt_buf, 0, p->idwt_stride * p->idwt_height * sizeof(IDWTELEM));

        }

        if (!s->zero_res) {

            if ((ret = decode_lowdelay(s)) < 0)

                return ret;

        }

    }



    for (comp = 0; comp < 3; comp++) {

        Plane *p       = &s->plane[comp];

        uint8_t *frame = s->current_picture->avframe->data[comp];



        /* FIXME: small resolutions */

        for (i = 0; i < 4; i++)

            s->edge_emu_buffer[i] = s->edge_emu_buffer_base + i*FFALIGN(p->width, 16);



        if (!s->zero_res && !s->low_delay)

        {

            memset(p->idwt_buf, 0, p->idwt_stride * p->idwt_height * sizeof(IDWTELEM));

            decode_component(s, comp); /* [DIRAC_STD] 13.4.1 core_transform_data() */

        }

        ret = ff_spatial_idwt_init2(&d, p->idwt_buf, p->idwt_width, p->idwt_height, p->idwt_stride,

                                    s->wavelet_idx+2, s->wavelet_depth, p->idwt_tmp);

        if (ret < 0)

            return ret;



        if (!s->num_refs) { /* intra */

            for (y = 0; y < p->height; y += 16) {

                ff_spatial_idwt_slice2(&d, y+16); /* decode */

                s->diracdsp.put_signed_rect_clamped(frame + y*p->stride, p->stride,

                                                    p->idwt_buf + y*p->idwt_stride, p->idwt_stride, p->width, 16);

            }

        } else { /* inter */

            int rowheight = p->ybsep*p->stride;



            select_dsp_funcs(s, p->width, p->height, p->xblen, p->yblen);



            for (i = 0; i < s->num_refs; i++)

                interpolate_refplane(s, s->ref_pics[i], comp, p->width, p->height);



            memset(s->mctmp, 0, 4*p->yoffset*p->stride);



            dsty = -p->yoffset;

            for (y = 0; y < s->blheight; y++) {

                int h     = 0,

                    start = FFMAX(dsty, 0);

                uint16_t *mctmp    = s->mctmp + y*rowheight;

                DiracBlock *blocks = s->blmotion + y*s->blwidth;



                init_obmc_weights(s, p, y);



                if (y == s->blheight-1 || start+p->ybsep > p->height)

                    h = p->height - start;

                else

                    h = p->ybsep - (start - dsty);

                if (h < 0)

                    break;



                memset(mctmp+2*p->yoffset*p->stride, 0, 2*rowheight);

                mc_row(s, blocks, mctmp, comp, dsty);



                mctmp += (start - dsty)*p->stride + p->xoffset;

                ff_spatial_idwt_slice2(&d, start + h); /* decode */

                s->diracdsp.add_rect_clamped(frame + start*p->stride, mctmp, p->stride,

                                             p->idwt_buf + start*p->idwt_stride, p->idwt_stride, p->width, h);



                dsty += p->ybsep;

            }

        }

    }





    return 0;

}
