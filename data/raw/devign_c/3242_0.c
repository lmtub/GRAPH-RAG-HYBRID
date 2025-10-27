void align_get_bits(GetBitContext *s)

{

    int n= (-get_bits_count(s)) & 7;

    if(n) skip_bits(s, n);

}
