float64 int32_to_float64( int32 a STATUS_PARAM )

{

    flag zSign;

    uint32 absA;

    int8 shiftCount;

    bits64 zSig;



    if ( a == 0 ) return 0;

    zSign = ( a < 0 );

    absA = zSign ? - a : a;

    shiftCount = countLeadingZeros32( absA ) + 21;

    zSig = absA;

    return packFloat64( zSign, 0x432 - shiftCount, zSig<<shiftCount );



}
