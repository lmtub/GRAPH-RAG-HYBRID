static void vc1_interp_mc(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    H264ChromaContext *h264chroma = &v->h264chroma;

    uint8_t *srcY, *srcU, *srcV;

    int dxy, mx, my, uvmx, uvmy, src_x, src_y, uvsrc_x, uvsrc_y;

    int off, off_uv;

    int v_edge_pos = s->v_edge_pos >> v->field_mode;

    int use_ic = v->next_use_ic;



    if (!v->field_mode && !v->s.next_picture.f.data[0])

        return;



    mx   = s->mv[1][0][0];

    my   = s->mv[1][0][1];

    uvmx = (mx + ((mx & 3) == 3)) >> 1;

    uvmy = (my + ((my & 3) == 3)) >> 1;

    if (v->field_mode) {

        if (v->cur_field_type != v->ref_field_type[1])

            my   = my   - 2 + 4 * v->cur_field_type;

            uvmy = uvmy - 2 + 4 * v->cur_field_type;

    }

    if (v->fastuvmc) {

        uvmx = uvmx + ((uvmx < 0) ? -(uvmx & 1) : (uvmx & 1));

        uvmy = uvmy + ((uvmy < 0) ? -(uvmy & 1) : (uvmy & 1));

    }

    srcY = s->next_picture.f.data[0];

    srcU = s->next_picture.f.data[1];

    srcV = s->next_picture.f.data[2];



    src_x   = s->mb_x * 16 + (mx   >> 2);

    src_y   = s->mb_y * 16 + (my   >> 2);

    uvsrc_x = s->mb_x *  8 + (uvmx >> 2);

    uvsrc_y = s->mb_y *  8 + (uvmy >> 2);



    if (v->profile != PROFILE_ADVANCED) {

        src_x   = av_clip(  src_x, -16, s->mb_width  * 16);

        src_y   = av_clip(  src_y, -16, s->mb_height * 16);

        uvsrc_x = av_clip(uvsrc_x,  -8, s->mb_width  *  8);

        uvsrc_y = av_clip(uvsrc_y,  -8, s->mb_height *  8);

    } else {

        src_x   = av_clip(  src_x, -17, s->avctx->coded_width);

        src_y   = av_clip(  src_y, -18, s->avctx->coded_height + 1);

        uvsrc_x = av_clip(uvsrc_x,  -8, s->avctx->coded_width  >> 1);

        uvsrc_y = av_clip(uvsrc_y,  -8, s->avctx->coded_height >> 1);

    }



    srcY += src_y   * s->linesize   + src_x;

    srcU += uvsrc_y * s->uvlinesize + uvsrc_x;

    srcV += uvsrc_y * s->uvlinesize + uvsrc_x;



    if (v->field_mode && v->ref_field_type[1]) {

        srcY += s->current_picture_ptr->f.linesize[0];

        srcU += s->current_picture_ptr->f.linesize[1];

        srcV += s->current_picture_ptr->f.linesize[2];

    }



    /* for grayscale we should not try to read from unknown area */

    if (s->flags & CODEC_FLAG_GRAY) {

        srcU = s->edge_emu_buffer + 18 * s->linesize;

        srcV = s->edge_emu_buffer + 18 * s->linesize;

    }



    if (v->rangeredfrm || s->h_edge_pos < 22 || v_edge_pos < 22 || use_ic

        || (unsigned)(src_x - 1) > s->h_edge_pos - (mx & 3) - 16 - 3

        || (unsigned)(src_y - 1) > v_edge_pos    - (my & 3) - 16 - 3) {

        uint8_t *uvbuf = s->edge_emu_buffer + 19 * s->linesize;



        srcY -= s->mspel * (1 + s->linesize);

        s->vdsp.emulated_edge_mc(s->edge_emu_buffer, srcY,

                                 s->linesize, s->linesize,

                                 17 + s->mspel * 2, 17 + s->mspel * 2,

                                 src_x - s->mspel, src_y - s->mspel,

                                 s->h_edge_pos, v_edge_pos);

        srcY = s->edge_emu_buffer;

        s->vdsp.emulated_edge_mc(uvbuf, srcU,

                                 s->uvlinesize, s->uvlinesize,

                                 8 + 1, 8 + 1,

                                 uvsrc_x, uvsrc_y, s->h_edge_pos >> 1, v_edge_pos >> 1);

        s->vdsp.emulated_edge_mc(uvbuf + 16, srcV,

                                 s->uvlinesize, s->uvlinesize,

                                 8 + 1, 8 + 1,

                                 uvsrc_x, uvsrc_y, s->h_edge_pos >> 1, v_edge_pos >> 1);

        srcU = uvbuf;

        srcV = uvbuf + 16;

        /* if we deal with range reduction we need to scale source blocks */

        if (v->rangeredfrm) {

            int i, j;

            uint8_t *src, *src2;



            src = srcY;

            for (j = 0; j < 17 + s->mspel * 2; j++) {

                for (i = 0; i < 17 + s->mspel * 2; i++)

                    src[i] = ((src[i] - 128) >> 1) + 128;

                src += s->linesize;

            }

            src = srcU;

            src2 = srcV;

            for (j = 0; j < 9; j++) {

                for (i = 0; i < 9; i++) {

                    src[i]  = ((src[i]  - 128) >> 1) + 128;

                    src2[i] = ((src2[i] - 128) >> 1) + 128;

                }

                src  += s->uvlinesize;

                src2 += s->uvlinesize;

            }

        }



        if (use_ic) {

            uint8_t (*luty )[256] = v->next_luty;

            uint8_t (*lutuv)[256] = v->next_lutuv;

            int i, j;

            uint8_t *src, *src2;



            src = srcY;

            for (j = 0; j < 17 + s->mspel * 2; j++) {

                int f = v->field_mode ? v->ref_field_type[1] : ((j+src_y - s->mspel) & 1);

                for (i = 0; i < 17 + s->mspel * 2; i++)

                    src[i] = luty[f][src[i]];

                src += s->linesize;

            }

            src  = srcU;

            src2 = srcV;

            for (j = 0; j < 9; j++) {

                int f = v->field_mode ? v->ref_field_type[1] : ((j+uvsrc_y) & 1);

                for (i = 0; i < 9; i++) {

                    src[i]  = lutuv[f][src[i]];

                    src2[i] = lutuv[f][src2[i]];

                }

                src  += s->uvlinesize;

                src2 += s->uvlinesize;

            }

        }

        srcY += s->mspel * (1 + s->linesize);

    }



    off    = 0;

    off_uv = 0;



    if (s->mspel) {

        dxy = ((my & 3) << 2) | (mx & 3);

        v->vc1dsp.avg_vc1_mspel_pixels_tab[dxy](s->dest[0] + off    , srcY    , s->linesize, v->rnd);

        v->vc1dsp.avg_vc1_mspel_pixels_tab[dxy](s->dest[0] + off + 8, srcY + 8, s->linesize, v->rnd);

        srcY += s->linesize * 8;

        v->vc1dsp.avg_vc1_mspel_pixels_tab[dxy](s->dest[0] + off + 8 * s->linesize    , srcY    , s->linesize, v->rnd);

        v->vc1dsp.avg_vc1_mspel_pixels_tab[dxy](s->dest[0] + off + 8 * s->linesize + 8, srcY + 8, s->linesize, v->rnd);

    } else { // hpel mc

        dxy = (my & 2) | ((mx & 2) >> 1);



        if (!v->rnd)

            s->hdsp.avg_pixels_tab[0][dxy](s->dest[0] + off, srcY, s->linesize, 16);

        else

            s->hdsp.avg_no_rnd_pixels_tab[dxy](s->dest[0] + off, srcY, s->linesize, 16);

    }



    if (s->flags & CODEC_FLAG_GRAY) return;

    /* Chroma MC always uses qpel blilinear */

    uvmx = (uvmx & 3) << 1;

    uvmy = (uvmy & 3) << 1;

    if (!v->rnd) {

        h264chroma->avg_h264_chroma_pixels_tab[0](s->dest[1] + off_uv, srcU, s->uvlinesize, 8, uvmx, uvmy);

        h264chroma->avg_h264_chroma_pixels_tab[0](s->dest[2] + off_uv, srcV, s->uvlinesize, 8, uvmx, uvmy);

    } else {

        v->vc1dsp.avg_no_rnd_vc1_chroma_pixels_tab[0](s->dest[1] + off_uv, srcU, s->uvlinesize, 8, uvmx, uvmy);

        v->vc1dsp.avg_no_rnd_vc1_chroma_pixels_tab[0](s->dest[2] + off_uv, srcV, s->uvlinesize, 8, uvmx, uvmy);

    }

}
