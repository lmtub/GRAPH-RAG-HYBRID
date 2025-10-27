int floatx80_is_nan( floatx80 a1 )

{

    floatx80u u;

    u.f = a1;

    return ( ( u.i.high & 0x7FFF ) == 0x7FFF ) && (bits64) ( u.i.low<<1 );

}
