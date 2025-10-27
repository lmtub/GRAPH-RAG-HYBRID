int float32_le_quiet( float32 a, float32 b STATUS_PARAM )

{

    flag aSign, bSign;



    if (    ( ( extractFloat32Exp( a ) == 0xFF ) && extractFloat32Frac( a ) )

         || ( ( extractFloat32Exp( b ) == 0xFF ) && extractFloat32Frac( b ) )

       ) {

        if ( float32_is_signaling_nan( a ) || float32_is_signaling_nan( b ) ) {

            float_raise( float_flag_invalid STATUS_VAR);

        }

        return 0;

    }

    aSign = extractFloat32Sign( a );

    bSign = extractFloat32Sign( b );

    if ( aSign != bSign ) return aSign || ( (bits32) ( ( a | b )<<1 ) == 0 );

    return ( a == b ) || ( aSign ^ ( a < b ) );



}
