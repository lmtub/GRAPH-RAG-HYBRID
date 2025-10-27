static inline void unpack_coeffs(SnowContext *s, SubBand *b, SubBand * parent, int orientation){

    const int w= b->width;

    const int h= b->height;

    int x,y;

    

    if(1){

        int run;

        x_and_coeff *xc= b->x_coeff;

        x_and_coeff *prev_xc= NULL;

        x_and_coeff *prev2_xc= xc;

        x_and_coeff *parent_xc= parent ? parent->x_coeff : NULL;

        x_and_coeff *prev_parent_xc= parent_xc;



        run= get_symbol2(&s->c, b->state[1], 3);

        for(y=0; y<h; y++){

            int v=0;

            int lt=0, t=0, rt=0;



            if(y && prev_xc->x == 0){

                rt= prev_xc->coeff;

            }

            for(x=0; x<w; x++){

                int p=0;

                const int l= v;

                

                lt= t; t= rt;



                if(y){

                    if(prev_xc->x <= x)

                        prev_xc++;

                    if(prev_xc->x == x + 1)

                        rt= prev_xc->coeff;

                    else

                        rt=0;

                }

                if(parent_xc){

                    if(x>>1 > parent_xc->x){

                        parent_xc++;

                    }

                    if(x>>1 == parent_xc->x){

                        p= parent_xc->coeff;

                    }

                }

                if(/*ll|*/l|lt|t|rt|p){

                    int context= av_log2(/*ABS(ll) + */3*(l>>1) + (lt>>1) + (t&~1) + (rt>>1) + (p>>1));



                    v=get_rac(&s->c, &b->state[0][context]);

                    if(v){

                        v= 2*(get_symbol2(&s->c, b->state[context + 2], context-4) + 1);

                        v+=get_rac(&s->c, &b->state[0][16 + 1 + 3 + quant3bA[l&0xFF] + 3*quant3bA[t&0xFF]]);

                        

                        xc->x=x;

                        (xc++)->coeff= v;

                    }

                }else{

                    if(!run){

                        run= get_symbol2(&s->c, b->state[1], 3);

                        v= 2*(get_symbol2(&s->c, b->state[0 + 2], 0-4) + 1);

                        v+=get_rac(&s->c, &b->state[0][16 + 1 + 3]);

                        

                        xc->x=x;

                        (xc++)->coeff= v;

                    }else{

                        int max_run;

                        run--;

                        v=0;



                        if(y) max_run= FFMIN(run, prev_xc->x - x - 2);

                        else  max_run= FFMIN(run, w-x-1);

                        if(parent_xc)

                            max_run= FFMIN(max_run, 2*parent_xc->x - x - 1);

                        x+= max_run;

                        run-= max_run;

                    }

                }

            }

            (xc++)->x= w+1; //end marker

            prev_xc= prev2_xc;

            prev2_xc= xc;

            

            if(parent_xc){

                if(y&1){

                    while(parent_xc->x != parent->width+1)

                        parent_xc++;

                    parent_xc++;

                    prev_parent_xc= parent_xc;

                }else{

                    parent_xc= prev_parent_xc;

                }

            }

        }



        (xc++)->x= w+1; //end marker

    }

}
