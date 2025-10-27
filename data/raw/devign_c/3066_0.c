static inline uint64_t ucf64_dtoi(float64 d)

{

    union {

        uint64_t i;

        float64 d;

    } v;



    v.d = d;

    return v.i;

}
