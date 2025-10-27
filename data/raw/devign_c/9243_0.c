static int decode_header(SnowContext *s){

    int plane_index, tmp;

    uint8_t kstate[32];



    memset(kstate, MID_STATE, sizeof(kstate));



    s->keyframe= get_rac(&s->c, kstate);

    if(s->keyframe || s->always_reset){

        reset_contexts(s);

        s->spatial_decomposition_type=

        s->qlog=

        s->qbias=

        s->mv_scale=

        s->block_max_depth= 0;

    }

    if(s->keyframe){

        s->version= get_symbol(&s->c, s->header_state, 0);

        if(s->version>0){

            av_log(s->avctx, AV_LOG_ERROR, "version %d not supported", s->version);

            return -1;

        }

        s->always_reset= get_rac(&s->c, s->header_state);

        s->temporal_decomposition_type= get_symbol(&s->c, s->header_state, 0);

        s->temporal_decomposition_count= get_symbol(&s->c, s->header_state, 0);

        s->spatial_decomposition_count= get_symbol(&s->c, s->header_state, 0);

        s->colorspace_type= get_symbol(&s->c, s->header_state, 0);

        s->chroma_h_shift= get_symbol(&s->c, s->header_state, 0);

        s->chroma_v_shift= get_symbol(&s->c, s->header_state, 0);

        s->spatial_scalability= get_rac(&s->c, s->header_state);

//        s->rate_scalability= get_rac(&s->c, s->header_state);

        tmp= get_symbol(&s->c, s->header_state, 0)+1;

        if(tmp < 1 || tmp > MAX_REF_FRAMES){

            av_log(s->avctx, AV_LOG_ERROR, "reference frame count is %d\n", tmp);

            return -1;

        }

        s->max_ref_frames= tmp;



        decode_qlogs(s);

    }



    if(!s->keyframe){

        if(get_rac(&s->c, s->header_state)){

            for(plane_index=0; plane_index<2; plane_index++){

                int htaps, i, sum=0;

                Plane *p= &s->plane[plane_index];

                p->diag_mc= get_rac(&s->c, s->header_state);

                htaps= get_symbol(&s->c, s->header_state, 0)*2 + 2;

                if((unsigned)htaps > HTAPS_MAX || htaps==0)

                    return -1;

                p->htaps= htaps;

                for(i= htaps/2; i; i--){

                    p->hcoeff[i]= get_symbol(&s->c, s->header_state, 0) * (1-2*(i&1));

                    sum += p->hcoeff[i];

                }

                p->hcoeff[0]= 32-sum;

            }

            s->plane[2].diag_mc= s->plane[1].diag_mc;

            s->plane[2].htaps  = s->plane[1].htaps;

            memcpy(s->plane[2].hcoeff, s->plane[1].hcoeff, sizeof(s->plane[1].hcoeff));

        }

        if(get_rac(&s->c, s->header_state)){

            s->spatial_decomposition_count= get_symbol(&s->c, s->header_state, 0);

            decode_qlogs(s);

        }

    }



    s->spatial_decomposition_type+= get_symbol(&s->c, s->header_state, 1);

    if(s->spatial_decomposition_type > 1){

        av_log(s->avctx, AV_LOG_ERROR, "spatial_decomposition_type %d not supported", s->spatial_decomposition_type);

        return -1;

    }



    s->qlog           += get_symbol(&s->c, s->header_state, 1);

    s->mv_scale       += get_symbol(&s->c, s->header_state, 1);

    s->qbias          += get_symbol(&s->c, s->header_state, 1);

    s->block_max_depth+= get_symbol(&s->c, s->header_state, 1);

    if(s->block_max_depth > 1 || s->block_max_depth < 0){

        av_log(s->avctx, AV_LOG_ERROR, "block_max_depth= %d is too large", s->block_max_depth);

        s->block_max_depth= 0;

        return -1;

    }



    return 0;

}
