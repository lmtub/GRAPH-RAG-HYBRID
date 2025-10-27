static void decode_mb_b(AVSContext *h, enum cavs_mb mb_type) {

    int block;

    enum cavs_sub_mb sub_type[4];

    int flags;



    ff_cavs_init_mb(h);



    /* reset all MVs */

    h->mv[MV_FWD_X0] = ff_cavs_dir_mv;

    set_mvs(&h->mv[MV_FWD_X0], BLK_16X16);

    h->mv[MV_BWD_X0] = ff_cavs_dir_mv;

    set_mvs(&h->mv[MV_BWD_X0], BLK_16X16);

    switch(mb_type) {

    case B_SKIP:

    case B_DIRECT:

        if(!h->col_type_base[h->mbidx]) {

            /* intra MB at co-location, do in-plane prediction */

            ff_cavs_mv(h, MV_FWD_X0, MV_FWD_C2, MV_PRED_BSKIP, BLK_16X16, 1);

            ff_cavs_mv(h, MV_BWD_X0, MV_BWD_C2, MV_PRED_BSKIP, BLK_16X16, 0);

        } else

            /* direct prediction from co-located P MB, block-wise */

            for(block=0;block<4;block++)

                mv_pred_direct(h,&h->mv[mv_scan[block]],

                                 &h->col_mv[h->mbidx*4 + block]);

        break;

    case B_FWD_16X16:

        ff_cavs_mv(h, MV_FWD_X0, MV_FWD_C2, MV_PRED_MEDIAN, BLK_16X16, 1);

        break;

    case B_SYM_16X16:

        ff_cavs_mv(h, MV_FWD_X0, MV_FWD_C2, MV_PRED_MEDIAN, BLK_16X16, 1);

        mv_pred_sym(h, &h->mv[MV_FWD_X0], BLK_16X16);

        break;

    case B_BWD_16X16:

        ff_cavs_mv(h, MV_BWD_X0, MV_BWD_C2, MV_PRED_MEDIAN, BLK_16X16, 0);

        break;

    case B_8X8:

        for(block=0;block<4;block++)

            sub_type[block] = get_bits(&h->s.gb,2);

        for(block=0;block<4;block++) {

            switch(sub_type[block]) {

            case B_SUB_DIRECT:

                if(!h->col_type_base[h->mbidx]) {

                    /* intra MB at co-location, do in-plane prediction */

                    ff_cavs_mv(h, mv_scan[block], mv_scan[block]-3,

                            MV_PRED_BSKIP, BLK_8X8, 1);

                    ff_cavs_mv(h, mv_scan[block]+MV_BWD_OFFS,

                            mv_scan[block]-3+MV_BWD_OFFS,

                            MV_PRED_BSKIP, BLK_8X8, 0);

                } else

                    mv_pred_direct(h,&h->mv[mv_scan[block]],

                                   &h->col_mv[h->mbidx*4 + block]);

                break;

            case B_SUB_FWD:

                ff_cavs_mv(h, mv_scan[block], mv_scan[block]-3,

                        MV_PRED_MEDIAN, BLK_8X8, 1);

                break;

            case B_SUB_SYM:

                ff_cavs_mv(h, mv_scan[block], mv_scan[block]-3,

                        MV_PRED_MEDIAN, BLK_8X8, 1);

                mv_pred_sym(h, &h->mv[mv_scan[block]], BLK_8X8);

                break;

            }

        }

        for(block=0;block<4;block++) {

            if(sub_type[block] == B_SUB_BWD)

                ff_cavs_mv(h, mv_scan[block]+MV_BWD_OFFS,

                        mv_scan[block]+MV_BWD_OFFS-3,

                        MV_PRED_MEDIAN, BLK_8X8, 0);

        }

        break;

    default:

        av_assert2((mb_type > B_SYM_16X16) && (mb_type < B_8X8));

        flags = ff_cavs_partition_flags[mb_type];

        if(mb_type & 1) { /* 16x8 macroblock types */

            if(flags & FWD0)

                ff_cavs_mv(h, MV_FWD_X0, MV_FWD_C2, MV_PRED_TOP,  BLK_16X8, 1);

            if(flags & SYM0)

                mv_pred_sym(h, &h->mv[MV_FWD_X0], BLK_16X8);

            if(flags & FWD1)

                ff_cavs_mv(h, MV_FWD_X2, MV_FWD_A1, MV_PRED_LEFT, BLK_16X8, 1);

            if(flags & SYM1)

                mv_pred_sym(h, &h->mv[MV_FWD_X2], BLK_16X8);

            if(flags & BWD0)

                ff_cavs_mv(h, MV_BWD_X0, MV_BWD_C2, MV_PRED_TOP,  BLK_16X8, 0);

            if(flags & BWD1)

                ff_cavs_mv(h, MV_BWD_X2, MV_BWD_A1, MV_PRED_LEFT, BLK_16X8, 0);

        } else {          /* 8x16 macroblock types */

            if(flags & FWD0)

                ff_cavs_mv(h, MV_FWD_X0, MV_FWD_B3, MV_PRED_LEFT, BLK_8X16, 1);

            if(flags & SYM0)

                mv_pred_sym(h, &h->mv[MV_FWD_X0], BLK_8X16);

            if(flags & FWD1)

                ff_cavs_mv(h,MV_FWD_X1,MV_FWD_C2,MV_PRED_TOPRIGHT,BLK_8X16,1);

            if(flags & SYM1)

                mv_pred_sym(h, &h->mv[MV_FWD_X1], BLK_8X16);

            if(flags & BWD0)

                ff_cavs_mv(h, MV_BWD_X0, MV_BWD_B3, MV_PRED_LEFT, BLK_8X16, 0);

            if(flags & BWD1)

                ff_cavs_mv(h,MV_BWD_X1,MV_BWD_C2,MV_PRED_TOPRIGHT,BLK_8X16,0);

        }

    }

    ff_cavs_inter(h, mb_type);

    set_intra_mode_default(h);

    if(mb_type != B_SKIP)

        decode_residual_inter(h);

    ff_cavs_filter(h,mb_type);

}
