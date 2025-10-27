static float32 addFloat32Sigs( float32 a, float32 b, flag zSign STATUS_PARAM)

{

    int16 aExp, bExp, zExp;

    uint32_t aSig, bSig, zSig;

    int16 expDiff;



    aSig = extractFloat32Frac( a );

    aExp = extractFloat32Exp( a );

    bSig = extractFloat32Frac( b );

    bExp = extractFloat32Exp( b );

    expDiff = aExp - bExp;

    aSig <<= 6;

    bSig <<= 6;

    if ( 0 < expDiff ) {

        if ( aExp == 0xFF ) {

            if ( aSig ) return propagateFloat32NaN( a, b STATUS_VAR );

            return a;

        }

        if ( bExp == 0 ) {

            --expDiff;

        }

        else {

            bSig |= 0x20000000;

        }

        shift32RightJamming( bSig, expDiff, &bSig );

        zExp = aExp;

    }

    else if ( expDiff < 0 ) {

        if ( bExp == 0xFF ) {

            if ( bSig ) return propagateFloat32NaN( a, b STATUS_VAR );

            return packFloat32( zSign, 0xFF, 0 );

        }

        if ( aExp == 0 ) {

            ++expDiff;

        }

        else {

            aSig |= 0x20000000;

        }

        shift32RightJamming( aSig, - expDiff, &aSig );

        zExp = bExp;

    }

    else {

        if ( aExp == 0xFF ) {

            if ( aSig | bSig ) return propagateFloat32NaN( a, b STATUS_VAR );

            return a;

        }

        if ( aExp == 0 ) {

            if ( STATUS(flush_to_zero) ) return packFloat32( zSign, 0, 0 );

            return packFloat32( zSign, 0, ( aSig + bSig )>>6 );

        }

        zSig = 0x40000000 + aSig + bSig;

        zExp = aExp;

        goto roundAndPack;

    }

    aSig |= 0x20000000;

    zSig = ( aSig + bSig )<<1;

    --zExp;

    if ( (int32_t) zSig < 0 ) {

        zSig = aSig + bSig;

        ++zExp;

    }

 roundAndPack:

    return roundAndPackFloat32( zSign, zExp, zSig STATUS_VAR );



}
