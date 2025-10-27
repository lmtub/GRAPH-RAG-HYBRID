int float32_is_nan( float32 a1 )

{

    float32u u;

    uint64_t a;

    u.f = a1;

    a = u.i;

    return ( 0xFF800000 < ( a<<1 ) );

}
