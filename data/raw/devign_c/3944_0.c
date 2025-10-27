int float32_eq_signaling( float32 a, float32 b STATUS_PARAM )

{



    if (    ( ( extractFloat32Exp( a ) == 0xFF ) && extractFloat32Frac( a ) )

         || ( ( extractFloat32Exp( b ) == 0xFF ) && extractFloat32Frac( b ) )

       ) {

        float_raise( float_flag_invalid STATUS_VAR);

        return 0;

    }

    return ( a == b ) || ( (bits32) ( ( a | b )<<1 ) == 0 );



}
