static inline int bidir_refine(MpegEncContext * s, int mb_x, int mb_y)

{

    MotionEstContext * const c= &s->me;

    const int mot_stride = s->mb_stride;

    const int xy = mb_y *mot_stride + mb_x;

    int fbmin;

    int pred_fx= s->b_bidir_forw_mv_table[xy-1][0];

    int pred_fy= s->b_bidir_forw_mv_table[xy-1][1];

    int pred_bx= s->b_bidir_back_mv_table[xy-1][0];

    int pred_by= s->b_bidir_back_mv_table[xy-1][1];

    int motion_fx= s->b_bidir_forw_mv_table[xy][0]= s->b_forw_mv_table[xy][0];

    int motion_fy= s->b_bidir_forw_mv_table[xy][1]= s->b_forw_mv_table[xy][1];

    int motion_bx= s->b_bidir_back_mv_table[xy][0]= s->b_back_mv_table[xy][0];

    int motion_by= s->b_bidir_back_mv_table[xy][1]= s->b_back_mv_table[xy][1];

    const int flags= c->sub_flags;

    const int qpel= flags&FLAG_QPEL;

    const int shift= 1+qpel;

    const int xmin= c->xmin<<shift;

    const int ymin= c->ymin<<shift;

    const int xmax= c->xmax<<shift;

    const int ymax= c->ymax<<shift;

    uint8_t map[8][8][8][8];



    memset(map,0,sizeof(map));

#define BIDIR_MAP(fx,fy,bx,by) \

    map[(motion_fx+fx)&7][(motion_fy+fy)&7][(motion_bx+bx)&7][(motion_by+by)&7]

    BIDIR_MAP(0,0,0,0) = 1;



    fbmin= check_bidir_mv(s, motion_fx, motion_fy,

                          motion_bx, motion_by,

                          pred_fx, pred_fy,

                          pred_bx, pred_by,

                          0, 16);



    if(s->avctx->bidir_refine){

        int score, end;

#define CHECK_BIDIR(fx,fy,bx,by)\

    if( !BIDIR_MAP(fx,fy,bx,by)\

       &&(fx<=0 || motion_fx+fx<=xmax) && (fy<=0 || motion_fy+fy<=ymax) && (bx<=0 || motion_bx+bx<=xmax) && (by<=0 || motion_by+by<=ymax)\

       &&(fx>=0 || motion_fx+fx>=xmin) && (fy>=0 || motion_fy+fy>=ymin) && (bx>=0 || motion_bx+bx>=xmin) && (by>=0 || motion_by+by>=ymin)){\

        BIDIR_MAP(fx,fy,bx,by) = 1;\

        score= check_bidir_mv(s, motion_fx+fx, motion_fy+fy, motion_bx+bx, motion_by+by, pred_fx, pred_fy, pred_bx, pred_by, 0, 16);\

        if(score < fbmin){\

            fbmin= score;\

            motion_fx+=fx;\

            motion_fy+=fy;\

            motion_bx+=bx;\

            motion_by+=by;\

            end=0;\

        }\

    }

#define CHECK_BIDIR2(a,b,c,d)\

CHECK_BIDIR(a,b,c,d)\

CHECK_BIDIR(-a,-b,-c,-d)



#define CHECK_BIDIRR(a,b,c,d)\

CHECK_BIDIR2(a,b,c,d)\

CHECK_BIDIR2(b,c,d,a)\

CHECK_BIDIR2(c,d,a,b)\

CHECK_BIDIR2(d,a,b,c)



        do{

            end=1;



            CHECK_BIDIRR( 0, 0, 0, 1)

            if(s->avctx->bidir_refine > 1){

                CHECK_BIDIRR( 0, 0, 1, 1)

                CHECK_BIDIR2( 0, 1, 0, 1)

                CHECK_BIDIR2( 1, 0, 1, 0)

                CHECK_BIDIRR( 0, 0,-1, 1)

                CHECK_BIDIR2( 0,-1, 0, 1)

                CHECK_BIDIR2(-1, 0, 1, 0)

                if(s->avctx->bidir_refine > 2){

                    CHECK_BIDIRR( 0, 1, 1, 1)

                    CHECK_BIDIRR( 0,-1, 1, 1)

                    CHECK_BIDIRR( 0, 1,-1, 1)

                    CHECK_BIDIRR( 0, 1, 1,-1)

                    if(s->avctx->bidir_refine > 3){

                        CHECK_BIDIR2( 1, 1, 1, 1)

                        CHECK_BIDIRR( 1, 1, 1,-1)

                        CHECK_BIDIR2( 1, 1,-1,-1)

                        CHECK_BIDIR2( 1,-1,-1, 1)

                        CHECK_BIDIR2( 1,-1, 1,-1)

                    }

                }

            }

        }while(!end);

    }



    s->b_bidir_forw_mv_table[xy][0]= motion_fx;

    s->b_bidir_forw_mv_table[xy][1]= motion_fy;

    s->b_bidir_back_mv_table[xy][0]= motion_bx;

    s->b_bidir_back_mv_table[xy][1]= motion_by;



    return fbmin;

}
