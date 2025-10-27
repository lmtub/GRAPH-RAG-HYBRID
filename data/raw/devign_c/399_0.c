int floatx80_eq(floatx80 a, floatx80 b, float_status *status)

{



    if (    (    ( extractFloatx80Exp( a ) == 0x7FFF )

              && (uint64_t) ( extractFloatx80Frac( a )<<1 ) )

         || (    ( extractFloatx80Exp( b ) == 0x7FFF )

              && (uint64_t) ( extractFloatx80Frac( b )<<1 ) )

       ) {

        float_raise(float_flag_invalid, status);

        return 0;

    }

    return

           ( a.low == b.low )

        && (    ( a.high == b.high )

             || (    ( a.low == 0 )

                  && ( (uint16_t) ( ( a.high | b.high )<<1 ) == 0 ) )

           );



}
