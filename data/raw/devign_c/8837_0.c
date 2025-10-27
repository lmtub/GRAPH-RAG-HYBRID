int floatx80_unordered(floatx80 a, floatx80 b, float_status *status)

{

    if (    (    ( extractFloatx80Exp( a ) == 0x7FFF )

              && (uint64_t) ( extractFloatx80Frac( a )<<1 ) )

         || (    ( extractFloatx80Exp( b ) == 0x7FFF )

              && (uint64_t) ( extractFloatx80Frac( b )<<1 ) )

       ) {

        float_raise(float_flag_invalid, status);

        return 1;

    }

    return 0;

}
