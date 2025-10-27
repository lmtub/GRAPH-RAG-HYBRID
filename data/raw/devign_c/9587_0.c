static inline void backup_mb_border(H264Context *h, uint8_t *src_y, uint8_t *src_cb, uint8_t *src_cr, int linesize, int uvlinesize, int simple){

    MpegEncContext * const s = &h->s;

    int i;

    int step    = 1;

    int offset  = 1;

    int uvoffset= 1;

    int top_idx = 1;

    int skiplast= 0;



    src_y  -=   linesize;

    src_cb -= uvlinesize;

    src_cr -= uvlinesize;



    if(!simple && FRAME_MBAFF){

        if(s->mb_y&1){

            offset  = MB_MBAFF ? 1 : 17;

            uvoffset= MB_MBAFF ? 1 : 9;

            if(!MB_MBAFF){

                *(uint64_t*)(h->top_borders[0][s->mb_x]+ 0)= *(uint64_t*)(src_y +  15*linesize);

                *(uint64_t*)(h->top_borders[0][s->mb_x]+ 8)= *(uint64_t*)(src_y +8+15*linesize);

                if(simple || !CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                    *(uint64_t*)(h->top_borders[0][s->mb_x]+16)= *(uint64_t*)(src_cb+7*uvlinesize);

                    *(uint64_t*)(h->top_borders[0][s->mb_x]+24)= *(uint64_t*)(src_cr+7*uvlinesize);

                }

            }

        }else{

            if(!MB_MBAFF){

                h->left_border[0]= h->top_borders[0][s->mb_x][15];

                if(simple || !CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

                    h->left_border[34   ]= h->top_borders[0][s->mb_x][16+7  ];

                    h->left_border[34+18]= h->top_borders[0][s->mb_x][16+8+7];

                }

                skiplast= 1;

            }

            offset  =

            uvoffset=

            top_idx = MB_MBAFF ? 0 : 1;

        }

        step= MB_MBAFF ? 2 : 1;

    }



    // There are two lines saved, the line above the the top macroblock of a pair,

    // and the line above the bottom macroblock

    h->left_border[offset]= h->top_borders[top_idx][s->mb_x][15];

    for(i=1; i<17 - skiplast; i++){

        h->left_border[offset+i*step]= src_y[15+i*  linesize];

    }



    *(uint64_t*)(h->top_borders[top_idx][s->mb_x]+0)= *(uint64_t*)(src_y +  16*linesize);

    *(uint64_t*)(h->top_borders[top_idx][s->mb_x]+8)= *(uint64_t*)(src_y +8+16*linesize);



    if(simple || !CONFIG_GRAY || !(s->flags&CODEC_FLAG_GRAY)){

        h->left_border[uvoffset+34   ]= h->top_borders[top_idx][s->mb_x][16+7];

        h->left_border[uvoffset+34+18]= h->top_borders[top_idx][s->mb_x][24+7];

        for(i=1; i<9 - skiplast; i++){

            h->left_border[uvoffset+34   +i*step]= src_cb[7+i*uvlinesize];

            h->left_border[uvoffset+34+18+i*step]= src_cr[7+i*uvlinesize];

        }

        *(uint64_t*)(h->top_borders[top_idx][s->mb_x]+16)= *(uint64_t*)(src_cb+8*uvlinesize);

        *(uint64_t*)(h->top_borders[top_idx][s->mb_x]+24)= *(uint64_t*)(src_cr+8*uvlinesize);

    }

}
