static void vc1_mc_4mv_chroma(VC1Context *v, int dir)

{

    MpegEncContext *s = &v->s;

    H264ChromaContext *h264chroma = &v->h264chroma;

    uint8_t *srcU, *srcV;

    int uvmx, uvmy, uvsrc_x, uvsrc_y;

    int k, tx = 0, ty = 0;

    int mvx[4], mvy[4], intra[4], mv_f[4];

    int valid_count;

    int chroma_ref_type = v->cur_field_type;

    int v_edge_pos = s->v_edge_pos >> v->field_mode;

    uint8_t (*lutuv)[256];

    int use_ic;



    if (!v->field_mode && !v->s.last_picture.f.data[0])

        return;

    if (s->flags & CODEC_FLAG_GRAY)

        return;



    for (k = 0; k < 4; k++) {

        mvx[k] = s->mv[dir][k][0];

        mvy[k] = s->mv[dir][k][1];

        intra[k] = v->mb_type[0][s->block_index[k]];

        if (v->field_mode)

            mv_f[k] = v->mv_f[dir][s->block_index[k] + v->blocks_off];

    }



    /* calculate chroma MV vector from four luma MVs */

    if (!v->field_mode || (v->field_mode && !v->numref)) {

        valid_count = get_chroma_mv(mvx, mvy, intra, 0, &tx, &ty);

        chroma_ref_type = v->reffield;

        if (!valid_count) {

            s->current_picture.motion_val[1][s->block_index[0] + v->blocks_off][0] = 0;

            s->current_picture.motion_val[1][s->block_index[0] + v->blocks_off][1] = 0;

            v->luma_mv[s->mb_x][0] = v->luma_mv[s->mb_x][1] = 0;

            return; //no need to do MC for intra blocks

        }

    } else {

        int dominant = 0;

        if (mv_f[0] + mv_f[1] + mv_f[2] + mv_f[3] > 2)

            dominant = 1;

        valid_count = get_chroma_mv(mvx, mvy, mv_f, dominant, &tx, &ty);

        if (dominant)

            chroma_ref_type = !v->cur_field_type;

    }

    if (v->field_mode && chroma_ref_type == 1 && v->cur_field_type == 1 && !v->s.last_picture.f.data[0])

        return;

    s->current_picture.motion_val[1][s->block_index[0] + v->blocks_off][0] = tx;

    s->current_picture.motion_val[1][s->block_index[0] + v->blocks_off][1] = ty;

    uvmx = (tx + ((tx & 3) == 3)) >> 1;

    uvmy = (ty + ((ty & 3) == 3)) >> 1;



    v->luma_mv[s->mb_x][0] = uvmx;

    v->luma_mv[s->mb_x][1] = uvmy;



    if (v->fastuvmc) {

        uvmx = uvmx + ((uvmx < 0) ? (uvmx & 1) : -(uvmx & 1));

        uvmy = uvmy + ((uvmy < 0) ? (uvmy & 1) : -(uvmy & 1));

    }

    // Field conversion bias

    if (v->cur_field_type != chroma_ref_type)

        uvmy += 2 - 4 * chroma_ref_type;



    uvsrc_x = s->mb_x * 8 + (uvmx >> 2);

    uvsrc_y = s->mb_y * 8 + (uvmy >> 2);



    if (v->profile != PROFILE_ADVANCED) {

        uvsrc_x = av_clip(uvsrc_x, -8, s->mb_width  * 8);

        uvsrc_y = av_clip(uvsrc_y, -8, s->mb_height * 8);

    } else {

        uvsrc_x = av_clip(uvsrc_x, -8, s->avctx->coded_width  >> 1);

        uvsrc_y = av_clip(uvsrc_y, -8, s->avctx->coded_height >> 1);

    }



    if (!dir) {

        if (v->field_mode && (v->cur_field_type != chroma_ref_type) && v->second_field) {

            srcU = s->current_picture.f.data[1];

            srcV = s->current_picture.f.data[2];

            lutuv = v->curr_lutuv;

            use_ic = v->curr_use_ic;

        } else {

            srcU = s->last_picture.f.data[1];

            srcV = s->last_picture.f.data[2];

            lutuv = v->last_lutuv;

            use_ic = v->last_use_ic;

        }

    } else {

        srcU = s->next_picture.f.data[1];

        srcV = s->next_picture.f.data[2];

        lutuv = v->next_lutuv;

        use_ic = v->next_use_ic;

    }



    if (!srcU) {

        av_log(v->s.avctx, AV_LOG_ERROR, "Referenced frame missing.\n");

        return;

    }



    srcU += uvsrc_y * s->uvlinesize + uvsrc_x;

    srcV += uvsrc_y * s->uvlinesize + uvsrc_x;



    if (v->field_mode) {

        if (chroma_ref_type) {

            srcU += s->current_picture_ptr->f.linesize[1];

            srcV += s->current_picture_ptr->f.linesize[2];

        }

    }



    if (v->rangeredfrm || use_ic

        || s->h_edge_pos < 18 || v_edge_pos < 18

        || (unsigned)uvsrc_x > (s->h_edge_pos >> 1) - 9

        || (unsigned)uvsrc_y > (v_edge_pos    >> 1) - 9) {

        s->vdsp.emulated_edge_mc(s->edge_emu_buffer, srcU,

                                 s->uvlinesize, s->uvlinesize,

                                 8 + 1, 8 + 1, uvsrc_x, uvsrc_y,

                                 s->h_edge_pos >> 1, v_edge_pos >> 1);

        s->vdsp.emulated_edge_mc(s->edge_emu_buffer + 16, srcV,

                                 s->uvlinesize, s->uvlinesize,

                                 8 + 1, 8 + 1, uvsrc_x, uvsrc_y,

                                 s->h_edge_pos >> 1, v_edge_pos >> 1);

        srcU = s->edge_emu_buffer;

        srcV = s->edge_emu_buffer + 16;



        /* if we deal with range reduction we need to scale source blocks */

        if (v->rangeredfrm) {

            int i, j;

            uint8_t *src, *src2;



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



            src  = srcU;

            src2 = srcV;

            for (j = 0; j < 9; j++) {

                int f = v->field_mode ? chroma_ref_type : ((j + uvsrc_y) & 1);

                for (i = 0; i < 9; i++) {

                    src[i]  = lutuv[f][src[i]];

                    src2[i] = lutuv[f][src2[i]];

                }

                src  += s->uvlinesize;

                src2 += s->uvlinesize;

            }

        }

    }



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
