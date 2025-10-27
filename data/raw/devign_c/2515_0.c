static inline float32 ucf64_itos(uint32_t i)

{

    union {

        uint32_t i;

        float32 s;

    } v;



    v.i = i;

    return v.s;

}
