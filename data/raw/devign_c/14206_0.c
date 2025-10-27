static int skip_check(MpegEncContext *s, Picture *p, Picture *ref){

    int x, y, plane;

    int score=0;

    int64_t score64=0;



    for(plane=0; plane<3; plane++){

        const int stride= p->linesize[plane];

        const int bw= plane ? 1 : 2;

        for(y=0; y<s->mb_height*bw; y++){

            for(x=0; x<s->mb_width*bw; x++){

                int v= s->dsp.frame_skip_cmp[1](s, p->data[plane] + 8*(x + y*stride), ref->data[plane] + 8*(x + y*stride), stride, 8);

                

                switch(s->avctx->frame_skip_exp){

                    case 0: score= FFMAX(score, v); break;

                    case 1: score+= ABS(v);break;

                    case 2: score+= v*v;break;

                    case 3: score64+= ABS(v*v*(int64_t)v);break;

                    case 4: score64+= v*v*(int64_t)(v*v);break;

                }

            }

        }

    }

    

    if(score) score64= score;



    if(score64 < s->avctx->frame_skip_threshold)

        return 1;

    if(score64 < ((s->avctx->frame_skip_factor * (int64_t)s->lambda)>>8))

        return 1;

    return 0;

}
