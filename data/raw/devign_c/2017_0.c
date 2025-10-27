static inline int get_symbol_inline(RangeCoder *c, uint8_t *state, int is_signed){

    if(get_rac(c, state+0))

        return 0;

    else{

        int i, e, a;

        e= 0;

        while(get_rac(c, state+1 + e) && e<9){ //1..10

            e++;

        }



        a= 1;

        for(i=e-1; i>=0; i--){

            a += a + get_rac(c, state+22 + i); //22..31

        }



        e= -(is_signed && get_rac(c, state+11 + e)); //11..21

        return (a^e)-e;

    }

}
