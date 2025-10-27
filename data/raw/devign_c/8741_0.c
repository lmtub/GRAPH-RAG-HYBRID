int ff_h264_check_intra4x4_pred_mode(H264Context *h){

    MpegEncContext * const s = &h->s;

    static const int8_t top [12]= {-1, 0,LEFT_DC_PRED,-1,-1,-1,-1,-1, 0};

    static const int8_t left[12]= { 0,-1, TOP_DC_PRED, 0,-1,-1,-1, 0,-1,DC_128_PRED};

    int i;



    if(!(h->top_samples_available&0x8000)){

        for(i=0; i<4; i++){

            int status= top[ h->intra4x4_pred_mode_cache[scan8[0] + i] ];

            if(status<0){

                av_log(h->s.avctx, AV_LOG_ERROR, "top block unavailable for requested intra4x4 mode %d at %d %d\n", status, s->mb_x, s->mb_y);

                return -1;

            } else if(status){

                h->intra4x4_pred_mode_cache[scan8[0] + i]= status;

            }

        }

    }



    if((h->left_samples_available&0x8888)!=0x8888){

        static const int mask[4]={0x8000,0x2000,0x80,0x20};

        for(i=0; i<4; i++){

            if(!(h->left_samples_available&mask[i])){

                int status= left[ h->intra4x4_pred_mode_cache[scan8[0] + 8*i] ];

                if(status<0){

                    av_log(h->s.avctx, AV_LOG_ERROR, "left block unavailable for requested intra4x4 mode %d at %d %d\n", status, s->mb_x, s->mb_y);

                    return -1;

                } else if(status){

                    h->intra4x4_pred_mode_cache[scan8[0] + 8*i]= status;

                }

            }

        }

    }



    return 0;

} //FIXME cleanup like ff_h264_check_intra_pred_mode
