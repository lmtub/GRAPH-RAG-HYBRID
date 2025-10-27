void put_string(PutBitContext * pbc, char *s)

{

    while(*s){

        put_bits(pbc, 8, *s);

        s++;

    }

    put_bits(pbc, 8, 0);

}
