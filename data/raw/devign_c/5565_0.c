int floatx80_lt(floatx80 a, floatx80 b, float_status *status)

{

    flag aSign, bSign;



    if (    (    ( extractFloatx80Exp( a ) == 0x7FFF )

              && (uint64_t) ( extractFloatx80Frac( a )<<1 ) )

         || (    ( extractFloatx80Exp( b ) == 0x7FFF )

              && (uint64_t) ( extractFloatx80Frac( b )<<1 ) )

       ) {

        float_raise(float_flag_invalid, status);

        return 0;

    }

    aSign = extractFloatx80Sign( a );

    bSign = extractFloatx80Sign( b );

    if ( aSign != bSign ) {

        return

               aSign

            && (    ( ( (uint16_t) ( ( a.high | b.high )<<1 ) ) | a.low | b.low )

                 != 0 );

    }

    return

          aSign ? lt128( b.high, b.low, a.high, a.low )

        : lt128( a.high, a.low, b.high, b.low );



}
