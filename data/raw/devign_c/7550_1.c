static float32 roundAndPackFloat32( flag zSign, int16 zExp, uint32_t zSig STATUS_PARAM)

{

    int8 roundingMode;

    flag roundNearestEven;

    int8 roundIncrement, roundBits;

    flag isTiny;



    roundingMode = STATUS(float_rounding_mode);

    roundNearestEven = ( roundingMode == float_round_nearest_even );

    roundIncrement = 0x40;

    if ( ! roundNearestEven ) {

        if ( roundingMode == float_round_to_zero ) {

            roundIncrement = 0;

        }

        else {

            roundIncrement = 0x7F;

            if ( zSign ) {

                if ( roundingMode == float_round_up ) roundIncrement = 0;

            }

            else {

                if ( roundingMode == float_round_down ) roundIncrement = 0;

            }

        }

    }

    roundBits = zSig & 0x7F;

    if ( 0xFD <= (uint16_t) zExp ) {

        if (    ( 0xFD < zExp )

             || (    ( zExp == 0xFD )

                  && ( (int32_t) ( zSig + roundIncrement ) < 0 ) )

           ) {

            float_raise( float_flag_overflow | float_flag_inexact STATUS_VAR);

            return packFloat32( zSign, 0xFF, - ( roundIncrement == 0 ));

        }

        if ( zExp < 0 ) {

            if ( STATUS(flush_to_zero) ) return packFloat32( zSign, 0, 0 );

            isTiny =

                   ( STATUS(float_detect_tininess) == float_tininess_before_rounding )

                || ( zExp < -1 )

                || ( zSig + roundIncrement < 0x80000000 );

            shift32RightJamming( zSig, - zExp, &zSig );

            zExp = 0;

            roundBits = zSig & 0x7F;

            if ( isTiny && roundBits ) float_raise( float_flag_underflow STATUS_VAR);

        }

    }

    if ( roundBits ) STATUS(float_exception_flags) |= float_flag_inexact;

    zSig = ( zSig + roundIncrement )>>7;

    zSig &= ~ ( ( ( roundBits ^ 0x40 ) == 0 ) & roundNearestEven );

    if ( zSig == 0 ) zExp = 0;

    return packFloat32( zSign, zExp, zSig );



}
