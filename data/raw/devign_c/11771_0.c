float32 float32_sqrt( float32 a STATUS_PARAM )

{

    flag aSign;

    int16 aExp, zExp;

    bits32 aSig, zSig;

    bits64 rem, term;



    aSig = extractFloat32Frac( a );

    aExp = extractFloat32Exp( a );

    aSign = extractFloat32Sign( a );

    if ( aExp == 0xFF ) {

        if ( aSig ) return propagateFloat32NaN( a, 0 STATUS_VAR );

        if ( ! aSign ) return a;

        float_raise( float_flag_invalid STATUS_VAR);

        return float32_default_nan;

    }

    if ( aSign ) {

        if ( ( aExp | aSig ) == 0 ) return a;

        float_raise( float_flag_invalid STATUS_VAR);

        return float32_default_nan;

    }

    if ( aExp == 0 ) {

        if ( aSig == 0 ) return 0;

        normalizeFloat32Subnormal( aSig, &aExp, &aSig );

    }

    zExp = ( ( aExp - 0x7F )>>1 ) + 0x7E;

    aSig = ( aSig | 0x00800000 )<<8;

    zSig = estimateSqrt32( aExp, aSig ) + 2;

    if ( ( zSig & 0x7F ) <= 5 ) {

        if ( zSig < 2 ) {

            zSig = 0x7FFFFFFF;

            goto roundAndPack;

        }

        aSig >>= aExp & 1;

        term = ( (bits64) zSig ) * zSig;

        rem = ( ( (bits64) aSig )<<32 ) - term;

        while ( (sbits64) rem < 0 ) {

            --zSig;

            rem += ( ( (bits64) zSig )<<1 ) | 1;

        }

        zSig |= ( rem != 0 );

    }

    shift32RightJamming( zSig, 1, &zSig );

 roundAndPack:

    return roundAndPackFloat32( 0, zExp, zSig STATUS_VAR );



}
