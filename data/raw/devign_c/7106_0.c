float64 float64_round_to_int( float64 a STATUS_PARAM )

{

    flag aSign;

    int16 aExp;

    bits64 lastBitMask, roundBitsMask;

    int8 roundingMode;

    float64 z;



    aExp = extractFloat64Exp( a );

    if ( 0x433 <= aExp ) {

        if ( ( aExp == 0x7FF ) && extractFloat64Frac( a ) ) {

            return propagateFloat64NaN( a, a STATUS_VAR );

        }

        return a;

    }

    if ( aExp < 0x3FF ) {

        if ( (bits64) ( a<<1 ) == 0 ) return a;

        STATUS(float_exception_flags) |= float_flag_inexact;

        aSign = extractFloat64Sign( a );

        switch ( STATUS(float_rounding_mode) ) {

         case float_round_nearest_even:

            if ( ( aExp == 0x3FE ) && extractFloat64Frac( a ) ) {

                return packFloat64( aSign, 0x3FF, 0 );

            }

            break;

         case float_round_down:

            return aSign ? LIT64( 0xBFF0000000000000 ) : 0;

         case float_round_up:

            return

            aSign ? LIT64( 0x8000000000000000 ) : LIT64( 0x3FF0000000000000 );

        }

        return packFloat64( aSign, 0, 0 );

    }

    lastBitMask = 1;

    lastBitMask <<= 0x433 - aExp;

    roundBitsMask = lastBitMask - 1;

    z = a;

    roundingMode = STATUS(float_rounding_mode);

    if ( roundingMode == float_round_nearest_even ) {

        z += lastBitMask>>1;

        if ( ( z & roundBitsMask ) == 0 ) z &= ~ lastBitMask;

    }

    else if ( roundingMode != float_round_to_zero ) {

        if ( extractFloat64Sign( z ) ^ ( roundingMode == float_round_up ) ) {

            z += roundBitsMask;

        }

    }

    z &= ~ roundBitsMask;

    if ( z != a ) STATUS(float_exception_flags) |= float_flag_inexact;

    return z;



}
