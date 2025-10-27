static float64 roundAndPackFloat64( flag zSign, int16 zExp, bits64 zSig STATUS_PARAM)

{

    int8 roundingMode;

    flag roundNearestEven;

    int16 roundIncrement, roundBits;

    flag isTiny;



    roundingMode = STATUS(float_rounding_mode);

    roundNearestEven = ( roundingMode == float_round_nearest_even );

    roundIncrement = 0x200;

    if ( ! roundNearestEven ) {

        if ( roundingMode == float_round_to_zero ) {

            roundIncrement = 0;

        }

        else {

            roundIncrement = 0x3FF;

            if ( zSign ) {

                if ( roundingMode == float_round_up ) roundIncrement = 0;

            }

            else {

                if ( roundingMode == float_round_down ) roundIncrement = 0;

            }

        }

    }

    roundBits = zSig & 0x3FF;

    if ( 0x7FD <= (bits16) zExp ) {

        if (    ( 0x7FD < zExp )

             || (    ( zExp == 0x7FD )

                  && ( (sbits64) ( zSig + roundIncrement ) < 0 ) )

           ) {

            float_raise( float_flag_overflow | float_flag_inexact STATUS_VAR);

            return packFloat64( zSign, 0x7FF, 0 ) - ( roundIncrement == 0 );

        }

        if ( zExp < 0 ) {

            isTiny =

                   ( STATUS(float_detect_tininess) == float_tininess_before_rounding )

                || ( zExp < -1 )

                || ( zSig + roundIncrement < LIT64( 0x8000000000000000 ) );

            shift64RightJamming( zSig, - zExp, &zSig );

            zExp = 0;

            roundBits = zSig & 0x3FF;

            if ( isTiny && roundBits ) float_raise( float_flag_underflow STATUS_VAR);

        }

    }

    if ( roundBits ) STATUS(float_exception_flags) |= float_flag_inexact;

    zSig = ( zSig + roundIncrement )>>10;

    zSig &= ~ ( ( ( roundBits ^ 0x200 ) == 0 ) & roundNearestEven );

    if ( zSig == 0 ) zExp = 0;

    return packFloat64( zSign, zExp, zSig );



}
