static uint64_t htonll(uint64_t v)

{

    union { uint32_t lv[2]; uint64_t llv; } u;

    u.lv[0] = htonl(v >> 32);

    u.lv[1] = htonl(v & 0xFFFFFFFFULL);

    return u.llv;

}
