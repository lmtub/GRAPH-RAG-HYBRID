static inline int decode_picture_parameter_set(H264Context *h, int bit_length){

    MpegEncContext * const s = &h->s;

    unsigned int pps_id= get_ue_golomb(&s->gb);

    PPS *pps;



    if(pps_id>=MAX_PPS_COUNT){

        av_log(h->s.avctx, AV_LOG_ERROR, "pps_id out of range\n");

        return -1;

    }

    pps = &h->pps_buffer[pps_id];



    pps->sps_id= get_ue_golomb(&s->gb);

    pps->cabac= get_bits1(&s->gb);

    pps->pic_order_present= get_bits1(&s->gb);

    pps->slice_group_count= get_ue_golomb(&s->gb) + 1;

    if(pps->slice_group_count > 1 ){

        pps->mb_slice_group_map_type= get_ue_golomb(&s->gb);

        av_log(h->s.avctx, AV_LOG_ERROR, "FMO not supported\n");

        switch(pps->mb_slice_group_map_type){

        case 0:

#if 0

|   for( i = 0; i <= num_slice_groups_minus1; i++ ) |   |        |

|    run_length[ i ]                                |1  |ue(v)   |

#endif

            break;

        case 2:

#if 0

|   for( i = 0; i < num_slice_groups_minus1; i++ )  |   |        |

|{                                                  |   |        |

|    top_left_mb[ i ]                               |1  |ue(v)   |

|    bottom_right_mb[ i ]                           |1  |ue(v)   |

|   }                                               |   |        |

#endif

            break;

        case 3:

        case 4:

        case 5:

#if 0

|   slice_group_change_direction_flag               |1  |u(1)    |

|   slice_group_change_rate_minus1                  |1  |ue(v)   |

#endif

            break;

        case 6:

#if 0

|   slice_group_id_cnt_minus1                       |1  |ue(v)   |

|   for( i = 0; i <= slice_group_id_cnt_minus1; i++ |   |        |

|)                                                  |   |        |

|    slice_group_id[ i ]                            |1  |u(v)    |

#endif

            break;

        }

    }

    pps->ref_count[0]= get_ue_golomb(&s->gb) + 1;

    pps->ref_count[1]= get_ue_golomb(&s->gb) + 1;

    if(pps->ref_count[0] > 32 || pps->ref_count[1] > 32){

        av_log(h->s.avctx, AV_LOG_ERROR, "reference overflow (pps)\n");

        return -1;

    }



    pps->weighted_pred= get_bits1(&s->gb);

    pps->weighted_bipred_idc= get_bits(&s->gb, 2);

    pps->init_qp= get_se_golomb(&s->gb) + 26;

    pps->init_qs= get_se_golomb(&s->gb) + 26;

    pps->chroma_qp_index_offset= get_se_golomb(&s->gb);

    pps->deblocking_filter_parameters_present= get_bits1(&s->gb);

    pps->constrained_intra_pred= get_bits1(&s->gb);

    pps->redundant_pic_cnt_present = get_bits1(&s->gb);



    pps->transform_8x8_mode= 0;

    h->dequant_coeff_pps= -1; //contents of sps/pps can change even if id doesn't, so reinit

    memset(pps->scaling_matrix4, 16, 6*16*sizeof(uint8_t));

    memset(pps->scaling_matrix8, 16, 2*64*sizeof(uint8_t));



    if(get_bits_count(&s->gb) < bit_length){

        pps->transform_8x8_mode= get_bits1(&s->gb);

        decode_scaling_matrices(h, &h->sps_buffer[pps->sps_id], pps, 0, pps->scaling_matrix4, pps->scaling_matrix8);

        get_se_golomb(&s->gb);  //second_chroma_qp_index_offset

    }



    if(s->avctx->debug&FF_DEBUG_PICT_INFO){

        av_log(h->s.avctx, AV_LOG_DEBUG, "pps:%u sps:%u %s slice_groups:%d ref:%d/%d %s qp:%d/%d/%d %s %s %s %s\n",

               pps_id, pps->sps_id,

               pps->cabac ? "CABAC" : "CAVLC",

               pps->slice_group_count,

               pps->ref_count[0], pps->ref_count[1],

               pps->weighted_pred ? "weighted" : "",

               pps->init_qp, pps->init_qs, pps->chroma_qp_index_offset,

               pps->deblocking_filter_parameters_present ? "LPAR" : "",

               pps->constrained_intra_pred ? "CONSTR" : "",

               pps->redundant_pic_cnt_present ? "REDU" : "",

               pps->transform_8x8_mode ? "8x8DCT" : ""

               );

    }



    return 0;

}
