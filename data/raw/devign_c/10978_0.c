INLINE int16 extractFloat32Exp( float32 a )

{



    return ( a>>23 ) & 0xFF;



}
