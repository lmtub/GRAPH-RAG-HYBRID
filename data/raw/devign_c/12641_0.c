int float64_le( float64 a, float64 b STATUS_PARAM )

{

    flag aSign, bSign;



    if (    ( ( extractFloat64Exp( a ) == 0x7FF ) && extractFloat64Frac( a ) )

         || ( ( extractFloat64Exp( b ) == 0x7FF ) && extractFloat64Frac( b ) )

       ) {

        float_raise( float_flag_invalid STATUS_VAR);

        return 0;

    }

    aSign = extractFloat64Sign( a );

    bSign = extractFloat64Sign( b );

    if ( aSign != bSign ) return aSign || ( (bits64) ( ( a | b )<<1 ) == 0 );

    return ( a == b ) || ( aSign ^ ( a < b ) );



}
