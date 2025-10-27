int64 float32_to_int64_round_to_zero( float32 a STATUS_PARAM )

{

    flag aSign;

    int16 aExp, shiftCount;

    bits32 aSig;

    bits64 aSig64;

    int64 z;



    aSig = extractFloat32Frac( a );

    aExp = extractFloat32Exp( a );

    aSign = extractFloat32Sign( a );

    shiftCount = aExp - 0xBE;

    if ( 0 <= shiftCount ) {

        if ( a != 0xDF000000 ) {

            float_raise( float_flag_invalid STATUS_VAR);

            if ( ! aSign || ( ( aExp == 0xFF ) && aSig ) ) {

                return LIT64( 0x7FFFFFFFFFFFFFFF );

            }

        }

        return (sbits64) LIT64( 0x8000000000000000 );

    }

    else if ( aExp <= 0x7E ) {

        if ( aExp | aSig ) STATUS(float_exception_flags) |= float_flag_inexact;

        return 0;

    }

    aSig64 = aSig | 0x00800000;

    aSig64 <<= 40;

    z = aSig64>>( - shiftCount );

    if ( (bits64) ( aSig64<<( shiftCount & 63 ) ) ) {

        STATUS(float_exception_flags) |= float_flag_inexact;

    }

    if ( aSign ) z = - z;

    return z;



}
