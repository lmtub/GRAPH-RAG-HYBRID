NEON_TYPE4(s8, int8_t)
NEON_TYPE4(u8, uint8_t)
NEON_TYPE2(s16, int16_t)
NEON_TYPE2(u16, uint16_t)
NEON_TYPE1(s32, int32_t)
NEON_TYPE1(u32, uint32_t)
#undef NEON_TYPE4
#undef NEON_TYPE2
#undef NEON_TYPE1
/* Copy from a uint32_t to a vector structure type.  */
#define NEON_UNPACK(vtype, dest, val) do { \
    union { \
        vtype v; \
        uint32_t i; \
    } conv_u; \
    conv_u.i = (val); \
    dest = conv_u.v; \
    } while(0)
/* Copy from a vector structure type to a uint32_t.  */
#define NEON_PACK(vtype, dest, val) do { \
    union { \
        vtype v; \
        uint32_t i; \
    } conv_u; \
    conv_u.v = (val); \
    dest = conv_u.i; \
    } while(0)
#define NEON_DO1 \
    NEON_FN(vdest.v1, vsrc1.v1, vsrc2.v1);
#define NEON_DO2 \
    NEON_FN(vdest.v1, vsrc1.v1, vsrc2.v1); \
    NEON_FN(vdest.v2, vsrc1.v2, vsrc2.v2);
#define NEON_DO4 \
    NEON_FN(vdest.v1, vsrc1.v1, vsrc2.v1); \
    NEON_FN(vdest.v2, vsrc1.v2, vsrc2.v2); \
    NEON_FN(vdest.v3, vsrc1.v3, vsrc2.v3); \
    NEON_FN(vdest.v4, vsrc1.v4, vsrc2.v4);
#define NEON_VOP_BODY(vtype, n) \
{ \
    uint32_t res; \
    vtype vsrc1; \
    vtype vsrc2; \
    vtype vdest; \
    NEON_UNPACK(vtype, vsrc1, arg1); \
    NEON_UNPACK(vtype, vsrc2, arg2); \
    NEON_DO##n; \
    NEON_PACK(vtype, res, vdest); \
    return res; \
#define NEON_VOP(name, vtype, n) \
uint32_t HELPER(glue(neon_,name))(uint32_t arg1, uint32_t arg2) \
NEON_VOP_BODY(vtype, n)
#define NEON_VOP_ENV(name, vtype, n) \
uint32_t HELPER(glue(neon_,name))(CPUState *env, uint32_t arg1, uint32_t arg2) \
NEON_VOP_BODY(vtype, n)
/* Pairwise operations.  */
/* For 32-bit elements each segment only contains a single element, so
   the elementwise and pairwise operations are the same.  */
#define NEON_PDO2 \
    NEON_FN(vdest.v1, vsrc1.v1, vsrc1.v2); \
    NEON_FN(vdest.v2, vsrc2.v1, vsrc2.v2);
#define NEON_PDO4 \
    NEON_FN(vdest.v1, vsrc1.v1, vsrc1.v2); \
    NEON_FN(vdest.v2, vsrc1.v3, vsrc1.v4); \
    NEON_FN(vdest.v3, vsrc2.v1, vsrc2.v2); \
    NEON_FN(vdest.v4, vsrc2.v3, vsrc2.v4); \
#define NEON_POP(name, vtype, n) \
uint32_t HELPER(glue(neon_,name))(uint32_t arg1, uint32_t arg2) \
{ \
    uint32_t res; \
    vtype vsrc1; \
    vtype vsrc2; \
    vtype vdest; \
    NEON_UNPACK(vtype, vsrc1, arg1); \
    NEON_UNPACK(vtype, vsrc2, arg2); \
    NEON_PDO##n; \
    NEON_PACK(vtype, res, vdest); \
    return res; \
/* Unary operators.  */
#define NEON_VOP1(name, vtype, n) \
uint32_t HELPER(glue(neon_,name))(uint32_t arg) \
{ \
    vtype vsrc1; \
    vtype vdest; \
    NEON_UNPACK(vtype, vsrc1, arg); \
    NEON_DO##n; \
    NEON_PACK(vtype, arg, vdest); \
    return arg; \
#define NEON_USAT(dest, src1, src2, type) do { \
    uint32_t tmp = (uint32_t)src1 + (uint32_t)src2; \
    if (tmp != (type)tmp) { \
        SET_QC(); \
        dest = ~0; \
    } else { \
        dest = tmp; \
    }} while(0)
#define NEON_FN(dest, src1, src2) NEON_USAT(dest, src1, src2, uint8_t)
NEON_VOP_ENV(qadd_u8, neon_u8, 4)
#undef NEON_FN
#define NEON_FN(dest, src1, src2) NEON_USAT(dest, src1, src2, uint16_t)
NEON_VOP_ENV(qadd_u16, neon_u16, 2)
#undef NEON_FN
#undef NEON_USAT