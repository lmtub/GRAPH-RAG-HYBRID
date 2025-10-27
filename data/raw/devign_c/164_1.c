void OPPROTO op_udiv_T1_T0(void)
{
    uint64_t x0;
    uint32_t x1;
    x0 = T0 | ((uint64_t) (env->y) << 32);
    x1 = T1;
    x0 = x0 / x1;
    if (x0 > 0xffffffff) {
	T0 = 0xffffffff;
	T1 = 1;
    } else {
	T0 = x0;
	T1 = 0;
    FORCE_RET();