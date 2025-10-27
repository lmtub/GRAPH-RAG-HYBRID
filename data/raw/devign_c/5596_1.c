static float128 addFloat128Sigs( float128 a, float128 b, flag zSign STATUS_PARAM)

{

    int32 aExp, bExp, zExp;

    uint64_t aSig0, aSig1, bSig0, bSig1, zSig0, zSig1, zSig2;

    int32 expDiff;



    aSig1 = extractFloat128Frac1( a );

    aSig0 = extractFloat128Frac0( a );

    aExp = extractFloat128Exp( a );

    bSig1 = extractFloat128Frac1( b );

    bSig0 = extractFloat128Frac0( b );

    bExp = extractFloat128Exp( b );

    expDiff = aExp - bExp;

    if ( 0 < expDiff ) {

        if ( aExp == 0x7FFF ) {

            if ( aSig0 | aSig1 ) return propagateFloat128NaN( a, b STATUS_VAR );

            return a;

        }

        if ( bExp == 0 ) {

            --expDiff;

        }

        else {

            bSig0 |= LIT64( 0x0001000000000000 );

        }

        shift128ExtraRightJamming(

            bSig0, bSig1, 0, expDiff, &bSig0, &bSig1, &zSig2 );

        zExp = aExp;

    }

    else if ( expDiff < 0 ) {

        if ( bExp == 0x7FFF ) {

            if ( bSig0 | bSig1 ) return propagateFloat128NaN( a, b STATUS_VAR );

            return packFloat128( zSign, 0x7FFF, 0, 0 );

        }

        if ( aExp == 0 ) {

            ++expDiff;

        }

        else {

            aSig0 |= LIT64( 0x0001000000000000 );

        }

        shift128ExtraRightJamming(

            aSig0, aSig1, 0, - expDiff, &aSig0, &aSig1, &zSig2 );

        zExp = bExp;

    }

    else {

        if ( aExp == 0x7FFF ) {

            if ( aSig0 | aSig1 | bSig0 | bSig1 ) {

                return propagateFloat128NaN( a, b STATUS_VAR );

            }

            return a;

        }

        add128( aSig0, aSig1, bSig0, bSig1, &zSig0, &zSig1 );

        if ( aExp == 0 ) {

            if ( STATUS(flush_to_zero) ) return packFloat128( zSign, 0, 0, 0 );

            return packFloat128( zSign, 0, zSig0, zSig1 );

        }

        zSig2 = 0;

        zSig0 |= LIT64( 0x0002000000000000 );

        zExp = aExp;

        goto shiftRight1;

    }

    aSig0 |= LIT64( 0x0001000000000000 );

    add128( aSig0, aSig1, bSig0, bSig1, &zSig0, &zSig1 );

    --zExp;

    if ( zSig0 < LIT64( 0x0002000000000000 ) ) goto roundAndPack;

    ++zExp;

 shiftRight1:

    shift128ExtraRightJamming(

        zSig0, zSig1, zSig2, 1, &zSig0, &zSig1, &zSig2 );

 roundAndPack:

    return roundAndPackFloat128( zSign, zExp, zSig0, zSig1, zSig2 STATUS_VAR );



}
