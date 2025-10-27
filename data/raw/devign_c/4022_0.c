static inline void put_symbol_inline(RangeCoder *c, uint8_t *state, int v, int is_signed){

    int i;



    if(v){

        const int a= FFABS(v);

        const int e= av_log2(a);

        put_rac(c, state+0, 0);



        assert(e<=9);



        for(i=0; i<e; i++){

            put_rac(c, state+1+i, 1);  //1..10

        }

        put_rac(c, state+1+i, 0);



        for(i=e-1; i>=0; i--){

            put_rac(c, state+22+i, (a>>i)&1); //22..31

        }



        if(is_signed)

            put_rac(c, state+11 + e, v < 0); //11..21

    }else{

        put_rac(c, state+0, 1);

    }

}
