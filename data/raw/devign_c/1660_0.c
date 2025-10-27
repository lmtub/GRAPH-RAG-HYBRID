static int encode_subband_c0run(SnowContext *s, SubBand *b, DWTELEM *src, DWTELEM *parent, int stride, int orientation){

    const int w= b->width;

    const int h= b->height;

    int x, y;



    if(1){

        int run=0;

        int runs[w*h];

        int run_index=0;

                

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int v, p=0;

                int /*ll=0, */l=0, lt=0, t=0, rt=0;

                v= src[x + y*stride];



                if(y){

                    t= src[x + (y-1)*stride];

                    if(x){

                        lt= src[x - 1 + (y-1)*stride];

                    }

                    if(x + 1 < w){

                        rt= src[x + 1 + (y-1)*stride];

                    }

                }

                if(x){

                    l= src[x - 1 + y*stride];

                    /*if(x > 1){

                        if(orientation==1) ll= src[y + (x-2)*stride];

                        else               ll= src[x - 2 + y*stride];

                    }*/

                }

                if(parent){

                    int px= x>>1;

                    int py= y>>1;

                    if(px<b->parent->width && py<b->parent->height) 

                        p= parent[px + py*2*stride];

                }

                if(!(/*ll|*/l|lt|t|rt|p)){

                    if(v){

                        runs[run_index++]= run;

                        run=0;

                    }else{

                        run++;

                    }

                }

            }

        }

        runs[run_index++]= run;

        run_index=0;

        run= runs[run_index++];



        put_symbol2(&s->c, b->state[1], run, 3);

        

        for(y=0; y<h; y++){

            if(s->c.bytestream_end - s->c.bytestream < w*40){

                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

                return -1;

            }

            for(x=0; x<w; x++){

                int v, p=0;

                int /*ll=0, */l=0, lt=0, t=0, rt=0;

                v= src[x + y*stride];



                if(y){

                    t= src[x + (y-1)*stride];

                    if(x){

                        lt= src[x - 1 + (y-1)*stride];

                    }

                    if(x + 1 < w){

                        rt= src[x + 1 + (y-1)*stride];

                    }

                }

                if(x){

                    l= src[x - 1 + y*stride];

                    /*if(x > 1){

                        if(orientation==1) ll= src[y + (x-2)*stride];

                        else               ll= src[x - 2 + y*stride];

                    }*/

                }

                if(parent){

                    int px= x>>1;

                    int py= y>>1;

                    if(px<b->parent->width && py<b->parent->height) 

                        p= parent[px + py*2*stride];

                }

                if(/*ll|*/l|lt|t|rt|p){

                    int context= av_log2(/*ABS(ll) + */3*ABS(l) + ABS(lt) + 2*ABS(t) + ABS(rt) + ABS(p));



                    put_rac(&s->c, &b->state[0][context], !!v);

                }else{

                    if(!run){

                        run= runs[run_index++];



                        put_symbol2(&s->c, b->state[1], run, 3);

                        assert(v);

                    }else{

                        run--;

                        assert(!v);

                    }

                }

                if(v){

                    int context= av_log2(/*ABS(ll) + */3*ABS(l) + ABS(lt) + 2*ABS(t) + ABS(rt) + ABS(p));

                    int l2= 2*ABS(l) + (l<0);

                    int t2= 2*ABS(t) + (t<0);



                    put_symbol2(&s->c, b->state[context + 2], ABS(v)-1, context-4);

                    put_rac(&s->c, &b->state[0][16 + 1 + 3 + quant3bA[l2&0xFF] + 3*quant3bA[t2&0xFF]], v<0);

                }

            }

        }

    }

    return 0;

}
