static void vc1_mc_1mv(VC1Context *v, int dir)

{

    MpegEncContext *s = &v->s;

    H264ChromaContext *h264chroma = &v->h264chroma;

    uint8_t *srcY, *srcU, *srcV;

    int dxy, mx, my, uvmx, uvmy, src_x, src_y, uvsrc_x, uvsrc_y;

    int v_edge_pos = s->v_edge_pos >> v->field_mode;

    int i;

    uint8_t (*luty)[256], (*lutuv)[256];

    int use_ic;



    if ((!v->field_mode ||

         (v->ref_field_type[dir] == 1 && v->cur_field_type == 1)) &&

        !v->s.last_picture.f.data[0])

        return;



    mx = s->mv[dir][0][0];

    my = s->mv[dir][0][1];



    // store motion vectors for further use in B frames

    if (s->pict_type == AV_PICTURE_TYPE_P) {

        for (i = 0; i < 4; i++) {

            s->current_picture.motion_val[1][s->block_index[i] + v->blocks_off][0] = mx;

            s->current_picture.motion_val[1][s->block_index[i] + v->blocks_off][1] = my;

        }

    }



    uvmx = (mx + ((mx & 3) == 3)) >> 1;

    uvmy = (my + ((my & 3) == 3)) >> 1;

    v->luma_mv[s->mb_x][0] = uvmx;

    v->luma_mv[s->mb_x][1] = uvmy;



    if (v->field_mode &&

        v->cur_field_type != v->ref_field_type[dir]) {

        my   = my   - 2 + 4 * v->cur_field_type;

        uvmy = uvmy - 2 + 4 * v->cur_field_type;

    }



    // fastuvmc shall be ignored for interlaced frame picture

    if (v->fastuvmc && (v->fcm != ILACE_FRAME)) {

        uvmx = uvmx + ((uvmx < 0) ? (uvmx & 1) : -(uvmx & 1));

        uvmy = uvmy + ((uvmy < 0) ? (uvmy & 1) : -(uvmy & 1));

    }

    if (!dir) {

        if (v->field_mode && (v->cur_field_type != v->ref_field_type[dir]) && v->second_field) {

            srcY = s->current_picture.f.data[0];

            srcU = s->current_picture.f.data[1];

            srcV = s->current_picture.f.data[2];

            luty  = v->curr_luty;

            lutuv = v->curr_lutuv;

            use_ic = v->curr_use_ic;

        } else {

            srcY = s->last_picture.f.data[0];

            srcU = s->last_picture.f.data[1];

            srcV = s->last_picture.f.data[2];

            luty  = v->last_luty;

            lutuv = v->last_lutuv;

            use_ic = v->last_use_ic;

        }

    } else {

        srcY = s->next_picture.f.data[0];

        srcU = s->next_picture.f.data[1];

        srcV = s->next_picture.f.data[2];

        luty  = v->next_luty;

        lutuv = v->next_lutuv;

        use_ic = v->next_use_ic;

    }



    if (!srcY || !srcU) {

        av_log(v->s.avctx, AV_LOG_ERROR, "Referenced frame missing.\n");

        return;

    }



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



    if (v->field_mode && v->ref_field_type[dir]) {

        srcY += s->current_picture_ptr->f.linesize[0];

        srcU += s->current_picture_ptr->f.linesize[1];

        srcV += s->current_picture_ptr->f.linesize[2];

    }



    /* for grayscale we should not try to read from unknown area */

    if (s->flags & CODEC_FLAG_GRAY) {

        srcU = s->edge_emu_buffer + 18 * s->linesize;

        srcV = s->edge_emu_buffer + 18 * s->linesize;

    }



    if (v->rangeredfrm || use_ic

        || s->h_edge_pos < 22 || v_edge_pos < 22

        || (unsigned)(src_x - s->mspel) > s->h_edge_pos - (mx&3) - 16 - s->mspel * 3

        || (unsigned)(src_y - 1)        > v_edge_pos    - (my&3) - 16 - 3) {

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

            src  = srcU;

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

        /* if we deal with intensity compensation we need to scale source blocks */

        if (use_ic) {

            int i, j;

            uint8_t *src, *src2;



            src = srcY;

            for (j = 0; j < 17 + s->mspel * 2; j++) {

                int f = v->field_mode ? v->ref_field_type[dir] : ((j + src_y - s->mspel) & 1) ;

                for (i = 0; i < 17 + s->mspel * 2; i++)

                    src[i] = luty[f][src[i]];

                src += s->linesize;

            }

            src  = srcU;

            src2 = srcV;

            for (j = 0; j < 9; j++) {

                int f = v->field_mode ? v->ref_field_type[dir] : ((j + uvsrc_y) & 1);

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



    if (s->mspel) {

        dxy = ((my & 3) << 2) | (mx & 3);

        v->vc1dsp.put_vc1_mspel_pixels_tab[dxy](s->dest[0]    , srcY    , s->linesize, v->rnd);

        v->vc1dsp.put_vc1_mspel_pixels_tab[dxy](s->dest[0] + 8, srcY + 8, s->linesize, v->rnd);

        srcY += s->linesize * 8;

        v->vc1dsp.put_vc1_mspel_pixels_tab[dxy](s->dest[0] + 8 * s->linesize    , srcY    , s->linesize, v->rnd);

        v->vc1dsp.put_vc1_mspel_pixels_tab[dxy](s->dest[0] + 8 * s->linesize + 8, srcY + 8, s->linesize, v->rnd);

    } else { // hpel mc - always used for luma

        dxy = (my & 2) | ((mx & 2) >> 1);

        if (!v->rnd)

            s->hdsp.put_pixels_tab[0][dxy](s->dest[0], srcY, s->linesize, 16);

        else

            s->hdsp.put_no_rnd_pixels_tab[0][dxy](s->dest[0], srcY, s->linesize, 16);

    }



    if (s->flags & CODEC_FLAG_GRAY) return;

    /* Chroma MC always uses qpel bilinear */

    uvmx = (uvmx & 3) << 1;

    uvmy = (uvmy & 3) << 1;

    if (!v->rnd) {

        h264chroma->put_h264_chroma_pixels_tab[0](s->dest[1], srcU, s->uvlinesize, 8, uvmx, uvmy);

        h264chroma->put_h264_chroma_pixels_tab[0](s->dest[2], srcV, s->uvlinesize, 8, uvmx, uvmy);

    } else {

        v->vc1dsp.put_no_rnd_vc1_chroma_pixels_tab[0](s->dest[1], srcU, s->uvlinesize, 8, uvmx, uvmy);

        v->vc1dsp.put_no_rnd_vc1_chroma_pixels_tab[0](s->dest[2], srcV, s->uvlinesize, 8, uvmx, uvmy);

    }

}
