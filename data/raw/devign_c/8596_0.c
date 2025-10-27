static int sab_diamond_search(MpegEncContext * s, int *best, int dmin,

                                       int src_index, int ref_index, int const penalty_factor,

                                       int size, int h, int flags)

{

    MotionEstContext * const c= &s->me;

    me_cmp_func cmpf, chroma_cmpf;

    Minima minima[MAX_SAB_SIZE];

    const int minima_count= FFABS(c->dia_size);

    int i, j;

    LOAD_COMMON

    LOAD_COMMON2

    unsigned map_generation = c->map_generation;



    av_assert1(minima_count <= MAX_SAB_SIZE);



    cmpf        = s->mecc.me_cmp[size];

    chroma_cmpf = s->mecc.me_cmp[size + 1];



    /*Note j<MAX_SAB_SIZE is needed if MAX_SAB_SIZE < ME_MAP_SIZE as j can

      become larger due to MVs overflowing their ME_MAP_MV_BITS bits space in map

     */

    for(j=i=0; i<ME_MAP_SIZE && j<MAX_SAB_SIZE; i++){

        uint32_t key= map[i];



        key += (1<<(ME_MAP_MV_BITS-1)) + (1<<(2*ME_MAP_MV_BITS-1));



        if((key&((-1)<<(2*ME_MAP_MV_BITS))) != map_generation) continue;



        minima[j].height= score_map[i];

        minima[j].x= key & ((1<<ME_MAP_MV_BITS)-1); key>>=ME_MAP_MV_BITS;

        minima[j].y= key & ((1<<ME_MAP_MV_BITS)-1);

        minima[j].x-= (1<<(ME_MAP_MV_BITS-1));

        minima[j].y-= (1<<(ME_MAP_MV_BITS-1));



        // all entries in map should be in range except if the mv overflows their ME_MAP_MV_BITS bits space

        if(   minima[j].x > xmax || minima[j].x < xmin

           || minima[j].y > ymax || minima[j].y < ymin)

            continue;



        minima[j].checked=0;

        if(minima[j].x || minima[j].y)

            minima[j].height+= (mv_penalty[((minima[j].x)<<shift)-pred_x] + mv_penalty[((minima[j].y)<<shift)-pred_y])*penalty_factor;



        j++;

    }



    qsort(minima, j, sizeof(Minima), minima_cmp);



    for(; j<minima_count; j++){

        minima[j].height=256*256*256*64;

        minima[j].checked=0;

        minima[j].x= minima[j].y=0;

    }



    for(i=0; i<minima_count; i++){

        const int x= minima[i].x;

        const int y= minima[i].y;

        int d;



        if(minima[i].checked) continue;



        if(   x >= xmax || x <= xmin

           || y >= ymax || y <= ymin)

           continue;



        SAB_CHECK_MV(x-1, y)

        SAB_CHECK_MV(x+1, y)

        SAB_CHECK_MV(x  , y-1)

        SAB_CHECK_MV(x  , y+1)



        minima[i].checked= 1;

    }



    best[0]= minima[0].x;

    best[1]= minima[0].y;

    dmin= minima[0].height;



    if(   best[0] < xmax && best[0] > xmin

       && best[1] < ymax && best[1] > ymin){

        int d;

        //ensure that the refernece samples for hpel refinement are in the map

        CHECK_MV(best[0]-1, best[1])

        CHECK_MV(best[0]+1, best[1])

        CHECK_MV(best[0], best[1]-1)

        CHECK_MV(best[0], best[1]+1)

    }

    return dmin;

}
